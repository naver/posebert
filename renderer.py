# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import pytorch3d
import pytorch3d.utils
import pytorch3d.renderer


class PyTorch3DRenderer(torch.nn.Module):
    """
    Thin wrapper around pytorch3d threed.
    Only square renderings are supported.
    Remark: PyTorch3D uses a camera convention with z going out of the camera and x pointing left.
    """

    def __init__(self,
                 image_size,
                 background_color=(0, 0, 0),
                 convention='opencv',
                 blur_radius=0,
                 faces_per_pixel=1,
                 bg_blending_radius=0,
                 ):
        super().__init__()
        self.image_size = image_size

        raster_settings_soft = pytorch3d.renderer.RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel)
        rasterizer = pytorch3d.renderer.MeshRasterizer(raster_settings=raster_settings_soft)

        materials = pytorch3d.renderer.materials.Materials(shininess=1.0)
        blend_params = pytorch3d.renderer.BlendParams(background_color=background_color)

        # One need to attribute a camera to the shader, otherwise the method "to" does not work.
        dummy_cameras = pytorch3d.renderer.OrthographicCameras()
        shader = pytorch3d.renderer.SoftPhongShader(cameras=dummy_cameras,
                                                    materials=materials,
                                                    blend_params=blend_params)

        # Differentiable soft threed using per vertex RGB colors for texture
        self.renderer = pytorch3d.renderer.MeshRenderer(rasterizer=rasterizer, shader=shader)

        self.convention = convention
        if convention == 'opencv':
            # Base camera rotation
            base_rotation = torch.as_tensor([[[-1, 0, 0],
                                              [0, -1, 0],
                                              [0, 0, 1]]], dtype=torch.float)
            self.register_buffer("base_rotation", base_rotation)
            self.register_buffer("base_rotation2d", base_rotation[:, 0:2, 0:2])

        # Light Color
        self.ambient_color = 0.5
        self.diffuse_color = 0.3
        self.specular_color = 0.2

        self.bg_blending_radius = bg_blending_radius
        if bg_blending_radius > 0:
            self.register_buffer("bg_blending_kernel",
                                 2.0 * torch.ones((1, 1, 2 * bg_blending_radius + 1, 2 * bg_blending_radius + 1)) / (
                                         2 * bg_blending_radius + 1) ** 2)
            self.register_buffer("bg_blending_bias", -torch.ones(1))
        else:
            self.blending_kernel = None
            self.blending_bias = None

    def to(self, device):
        # Transfer to device is a bit bugged in pytorch3d, one needs to do this manually
        self.renderer.shader.to(device)
        return super().to(device)

    def render(self, vertices, faces, cameras, color=None):
        """
        Args:
            - vertices: [B,N,V,3]
            - faces: [B,F,3]
            - maps: [B,N,W,H,3] in 0-1 range - if None the texture will be metallic
            - cameras: PerspectiveCamera or OrthographicCamera object
            - color: [B,N,3]
        Return:
            - img: [B,W,H,C]
        """
        K = vertices.size(2)
        N = vertices.size(1)
        list_faces = []
        list_vertices = []
        for i in range(N):
            list_faces.append(faces + K * i)
            list_vertices.append(vertices[:, i])
        faces = torch.cat(list_faces, 1)  # [B,N*13776,3]
        vertices = torch.cat(list_vertices, 1)  # [B,N*V,3]

        # Metallic texture
        verts_rgb = torch.ones_like(vertices).reshape(-1, N, K, 3)  # [1,N,V,3]
        if color is not None:
            verts_rgb = color.unsqueeze(2) * verts_rgb
        verts_rgb = verts_rgb.flatten(1, 2)
        textures = pytorch3d.renderer.Textures(verts_rgb=verts_rgb)

        # Create meshes
        meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)

        # Create light
        lights = pytorch3d.renderer.DirectionalLights(
            ambient_color=((self.ambient_color, self.ambient_color, self.ambient_color),),
            diffuse_color=((self.diffuse_color, self.diffuse_color, self.diffuse_color),),
            specular_color=(
                (self.specular_color, self.specular_color, self.specular_color),),
            direction=((0, 0, -1.0),),
            device=vertices.device)

        images = self.renderer(meshes, cameras=cameras, lights=lights)

        rgb_images = images[..., :3]
        rgb_images = torch.clamp(rgb_images, 0., 1.)
        rgb_images = rgb_images * 255
        rgb_images = rgb_images.to(torch.uint8)

        return rgb_images

    def renderPerspective(self, vertices, faces, camera_translation, principal_point=None, color=None, rotation=None,
                          focal_length=4):
        """
        Args:
            - vertices: [B,V,3] or [B,N,V,3] where N is the number of persons
            - faces: [B,13776,3]
            - focal_length: float
            - principal_point: [B,2]
            - T: [B,3]
            - color: [B,N,3]
        Return:
            - img: [B,W,H,C] in range 0-1
        """

        assert vertices.size(0) == faces.size(0) == camera_translation.size(0)
        batch_size = vertices.size(0)
        device = vertices.device

        if principal_point is None:
            principal_point = torch.zeros_like(camera_translation[:, :2])

        if vertices.dim() == 3:
            vertices = vertices.unsqueeze(1)
            if color is not None:
                assert color.dim() == 2
                color = color.unsqueeze(1)

        # Create cameras
        if rotation is None:
            R = self.base_rotation.repeat(batch_size, 1, 1)
        else:
            assert batch_size == rotation.size(0)
            R = torch.bmm(self.base_rotation.repeat(batch_size, 1, 1), rotation)
        camera_translation = torch.einsum('bik, bk -> bi', self.base_rotation.repeat(camera_translation.size(0), 1, 1),
                                          camera_translation)
        if self.convention == 'opencv':
            principal_point = -torch.as_tensor(principal_point)
        cameras = pytorch3d.renderer.PerspectiveCameras(focal_length=focal_length, principal_point=principal_point,
                                                        R=R, T=camera_translation, device=device)
        rgb_images = self.render(vertices, faces, cameras, color)

        return rgb_images

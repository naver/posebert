# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import cv2
import sys
from PIL import Image
import os
import torch
import numpy as np
import time
from torchvision import transforms
from smplx import SMPL
from pytorch3d.renderer import look_at_view_transform
import roma
import argparse

from dope import dope_resnet50, DOPE_NMS
from mocap_spin import HMR
from renderer import PyTorch3DRenderer
from posebert import PoseBERT
from skeleton import (convert_jts, get_h36m_traversal, normalize_skeleton_by_bone_length, preprocess_skeleton)


def main(videoname="fab_0.mp4", methodname="dope_posebert", sideview=False):
    assert os.path.isfile(videoname)
    assert methodname in ["mocapspin", "mocapspin_posebert", "dope_posebert"]

    models_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'models')
    color = [135, 206, 205]

    run_posebert = True if "posebert" in methodname else False
    run_mocapspin = True if "mocapspin" in methodname else False

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("\nLoading models...")
    # Load DOPE model
    ckpt_dope = torch.load(os.path.join(models_dir, 'DOPErealtime_v1_0_0.pth.tgz'), map_location=device)
    ckpt_dope['half'] = True
    dope = dope_resnet50(**ckpt_dope['dope_kwargs'])
    dope.eval()
    dope.load_state_dict(ckpt_dope['state_dict'])
    dope = dope.to(device)

    # Load MoCap-SPIN
    factor = 1.5
    ckpt_mocapSpin = torch.load(os.path.join(models_dir, 'mocapSPIN.pt'), map_location=device)
    mocap_spin = HMR()
    mocap_spin.eval()
    mocap_spin.load_state_dict(ckpt_mocapSpin['model'], strict=True)
    mocap_spin = mocap_spin.to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Load PoseBERT
    if run_mocapspin:
        ckpt_posebert_fn = os.path.join(models_dir, 'posebert_smpl.pt')
        in_dim = 24 * 6
    else:
        ckpt_posebert_fn = os.path.join(models_dir, 'posebert_h36m.pt')
        in_dim = 16 * 3 + 6
    ckpt_posebert = torch.load(ckpt_posebert_fn, map_location=device)
    init_pose = torch.from_numpy(np.load(os.path.join(models_dir, 'smpl_mean_params.npz'))['pose'][:]).unsqueeze(0)
    posebert = PoseBERT(init_pose=init_pose, in_dim=in_dim)
    posebert = posebert.eval()
    poserbert = posebert.to(device)
    posebert_seq_len = 64
    posebert.load_state_dict(ckpt_posebert['model_state_dict'])

    # Input video
    print(f"\nVideo: {videoname}")
    cap = cv2.VideoCapture(videoname)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Rendering
    bodymodel = SMPL(os.path.join(models_dir, "SMPL_NEUTRAL.pkl")).to(device)
    faces = torch.from_numpy(np.asarray(bodymodel.faces).astype(np.float32)).unsqueeze(0)
    renderer = PyTorch3DRenderer(height).to(device)
    dist, elev, azim = 5.5, 0., 180
    rotation, camera_translation = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    camera_translation[0, 0], camera_translation[0, 1] = 0., 0.2
    focal_length = 4.5

    # Mean 3d pose
    vertices_mean = bodymodel().vertices.detach()[0].cpu()
    J = torch.from_numpy(np.load(os.path.join(models_dir, "J_regressor_h36m.npy"))).float()
    h36m_mean = torch.matmul(J, vertices_mean).numpy()
    traversal, parents = get_h36m_traversal()

    # Read frame by frame
    print(f"\nRunning image-based methods...")
    list_mask, list_rotmat, list_betas, list_h36m = [], [], [], []
    list_duration_dope, list_duration_mocapspin, list_duration_reading = [], [], []
    t = 0
    while cap.isOpened():
        if t % int(cap.get(cv2.CAP_PROP_FPS)) == 0 and t > 0:
            print(f"frame: {t + 1:05d} - "
                  f"Reading: {np.asarray(list_duration_reading).mean() * 1000.:.1f} ms - "
                  f"DOPE: {np.asarray(list_duration_dope).mean()* 1000.:.1f} ms - "
                  f"MoCap-SPIN: {np.asarray(list_duration_mocapspin).mean()* 1000.:.1f} ms")
        t += 1
        sys.stdout.flush()

        start_time_reading = time.time()
        ret, img = cap.read()
        list_duration_reading.append(time.time() - start_time_reading)
        if ret == False:
            break

        #################################
        ### Detect a person with DOPE ###
        #################################
        start_time_dope = time.time()
        img = img[..., ::-1]  # BRG to RGB - [height,width,3]
        imlist = [torch.from_numpy(np.asarray(img) / 255.).permute(2, 0, 1).to(device).float()]
        resolution = imlist[0].size()[-2:]
        with torch.no_grad():
            results = dope(imlist, None)[0]
        parts = ['body', 'hand', 'face']
        
        detections, part = {}, 'body'
        dets, indices, bestcls = DOPE_NMS(results[part+'_scores'], results['boxes'], results[part+'_pose2d'], results[part+'_pose3d'], min_score=0.3)
        dets = {k: v.float().data.cpu().numpy() for k,v in dets.items()}
        detections[part] = [{'score': dets['score'][i], 'pose2d': dets['pose2d'][i,...], 'pose3d': dets['pose3d'][i,...]} for i in range(dets['score'].size)]

        duration_mocapspin = 0.
        if len(detections['body']) > 0:
            list_mask.append(True)

            ## Post-process DOPE ##
            # take the highest bbox
            size = [k['pose2d'][:, 1].max() - k['pose2d'][:, 1].min() for k in detections['body']]
            idx = size.index(max(size))

            # 3d pose estimated
            j3d = detections['body'][idx]['pose3d'] * np.asarray([[-1., -1., 1]])
            j3d = convert_jts(j3d.reshape(1, 13, 3), 'dope', 'h36m')
            j3d = j3d - j3d[:, [0]]  # center around the hip
            j3d = normalize_skeleton_by_bone_length(j3d[0], h36m_mean, traversal, parents).reshape(1, 17, 3)
            j3d_rel, _, j3d_root = preprocess_skeleton(j3d, center_joint=[0], xaxis=[1, 4], yaxis=[7, 0],
                                                       iter=5, norm_x_axis=True, norm_y_axis=True, sanity_check=False)
            list_h36m.append((torch.from_numpy(j3d_root).float(), torch.from_numpy(j3d_rel).float()))
            duration_dope = time.time() - start_time_dope

            ######################
            ### Run MoCap-SPIN ###
            ######################
            rotmat = torch.zeros((1, 24, 3, 3)).float()
            if run_mocapspin:
                start_time_mocapspin = time.time()
                # squared bbox
                j2d = detections['body'][idx]['pose2d']
                c_x = int((j2d[:, 0].min() + j2d[:, 0].max()) / 2.)
                c_y = int((j2d[:, 1].min() + j2d[:, 1].max()) / 2.)
                scale_x = int(factor * (j2d[:, 0].max() - j2d[:, 0].min()))
                scale_y = int(factor * (j2d[:, 1].max() - j2d[:, 1].min()))
                scale = max([scale_x, scale_y])
                left, top = int(c_x - scale / 2.), int(c_y - scale / 2.)
                right, bottom = left + scale, top + scale

                # crop, resize and estimate SMPL param with mocapSPIN
                img_crop = Image.fromarray(img).crop((left, top, right, bottom)).resize((224, 224))
                x = transform(img_crop).to(device)
                with torch.no_grad():
                    rotmat, betas, camera = mocap_spin(x.unsqueeze(0))
                rotmat = rotmat.float()
                list_betas.append(betas.cpu())
                duration_mocapspin = time.time() - start_time_mocapspin
            list_rotmat.append(rotmat.cpu())
        else:
            list_rotmat.append(torch.zeros((1, 24, 3, 3)).float())
            list_h36m.append((torch.zeros((1, 3, 3)).float(), torch.zeros((1, 17, 3)).float()))
            list_mask.append(False)
            duration_dope = time.time() - start_time_dope

        # Save time
        list_duration_dope.append(duration_dope)
        list_duration_mocapspin.append(duration_mocapspin)

    cap.release()
    cv2.destroyAllWindows()

    # Rendering and run PoseBERT
    print("\nRunning a video-based method (if posebert) and rendering the human mesh")
    betas = torch.zeros(1, 10).float() if len(list_betas) == 0 else torch.cat(list_betas).mean(0, keepdims=True)
    betas = betas.to(device)
    cap = cv2.VideoCapture(videoname)
    outfn = videoname + f"_{methodname}.mp4"
    wri = cv2.VideoWriter(outfn, cv2.VideoWriter_fourcc(*'mp4v'),
                          cap.get(cv2.CAP_PROP_FPS), (width + 2 * height, height))
    t = 0
    list_duration_posebert, list_duration_rendering, list_duration_smpl = [], [], []
    list_duration_reading, list_duration_saving = [], []
    while cap.isOpened():
        if t % int(cap.get(cv2.CAP_PROP_FPS)) == 0 and t > 0:
            print(f"frame: {t + 1:05d} - "
                  f"Reading: {np.asarray(list_duration_reading).mean()* 1000.:.1f} ms - "
                  f"PoseBERT: {np.asarray(list_duration_posebert).mean()* 1000.:.1f} ms - "
                  f"SMPL: {np.asarray(list_duration_smpl).mean()* 1000.:.1f} ms - "
                  f"Rendering: {np.asarray(list_duration_rendering).mean()* 1000.:.1f} ms - "
                  f"Saving: {np.asarray(list_duration_saving).mean()* 1000.:.1f} ms"
                  )
        sys.stdout.flush()

        # Reload the image
        start_time_reading = time.time()
        ret, img = cap.read()
        list_duration_reading.append(time.time() - start_time_reading)
        if ret == False:
            break

        ################
        ### PoseBERT ###
        ################
        rotmat = None
        duration_posebert = 0.
        if run_posebert:
            start_time_posebert = time.time()
            # select appropriate subseq
            if t - posebert_seq_len // 2 < 0:
                start_ = max([0, t - posebert_seq_len // 2])
                end_ = start_ + posebert_seq_len
            else:
                end_ = min([len(list_rotmat), t + posebert_seq_len // 2])
                start_ = end_ - posebert_seq_len
            tt = np.clip(np.arange(start_, end_), 0, len(list_rotmat) - 1).tolist()
            t_of_interest = tt.index(t)

            # PoseBERT
            rotmat = torch.cat([list_rotmat[j] for j in tt]).to(device)
            mask = torch.from_numpy(np.stack([list_mask[j] for j in tt])).bool().to(device)
            root = torch.cat([list_h36m[j][0] for j in tt]).to(device)
            rel = torch.cat([list_h36m[j][1] for j in tt]).to(device)
            with torch.no_grad():
                rotmat = poserbert(root=root.unsqueeze(0), rel=rel.unsqueeze(0), rotmat=rotmat.unsqueeze(0),
                                   mask=mask.unsqueeze(0))
            rotmat = rotmat[:, t_of_interest]
            duration_posebert = time.time() - start_time_posebert
        else:
            # no posebert on top, we keep the initial mocapspin pred
            if list_mask[t]:
                rotmat = list_rotmat[t].to(device)
        list_duration_posebert.append(duration_posebert)

        imout_mesh = np.zeros((renderer.image_size, renderer.image_size * 2, 3)).astype(np.uint8)
        duration_smpl, duration_rendering = 0., 0.
        if rotmat is not None:
            # SMPL layer
            start_time_smpl = time.time()
            rotvec = roma.rotmat_to_rotvec(rotmat)
            list_rotvec = [rotvec]
            if sideview:
                root_sideview = roma.rotvec_composition(
                    [rotvec[:, :1], torch.Tensor([[[0., np.pi / .4, 0.]]]).to(device)])
                rotvec_sideview = torch.cat([root_sideview, rotvec[:, 1:]], 1)
                list_rotvec.append(rotvec_sideview)
            rotvec_ = torch.cat(list_rotvec)
            with torch.no_grad():
                body = bodymodel(global_orient=rotvec_[:, 0], body_pose=rotvec_[:, 1:].flatten(1), betas=betas)
            vertices = body.vertices
            duration_smpl = time.time() - start_time_smpl

            # Rendering
            start_time_rendering = time.time()
            rep = vertices.size(0)
            with torch.no_grad():
                imout_mesh = renderer.renderPerspective(vertices=vertices,
                                                        camera_translation=camera_translation.to(device).repeat(rep, 1),
                                                        faces=faces.to(device).repeat(rep, 1, 1),
                                                        focal_length=focal_length,
                                                        rotation=rotation.to(device).repeat(rep, 1, 1),
                                                        color=torch.Tensor([[x / 255. for x in color]]).float().to(
                                                            device).repeat(rep, 1)
                                                        ).cpu().numpy()
                duration_rendering = time.time() - start_time_rendering
        list_duration_smpl.append(duration_smpl)
        list_duration_rendering.append(duration_rendering)

        start_time_saving = time.time()
        wri.write(np.concatenate([img] + [y[..., ::-1] for y in imout_mesh], 1))
        list_duration_saving.append(time.time() - start_time_saving)

        t += 1

    cap.release()
    wri.release()
    cv2.destroyAllWindows()

    # Summary of the computational time
    print(f"\n# Computational time #")
    list_duration_method = [
        list_duration_dope[t] + list_duration_mocapspin[t] + list_duration_posebert[t] + list_duration_smpl[t] for t in
        range(len(list_duration_dope))]
    list_duration_method_w_rendering = [
        list_duration_dope[t] + list_duration_mocapspin[t] + list_duration_posebert[t] + list_duration_smpl[t] +
        list_duration_rendering[t] for t in
        range(len(list_duration_dope))]
    print(f"{methodname}: {1 / np.asarray(list_duration_method).mean():.1f} fps")
    print(
        f"{methodname}+rendering(sideview={sideview}): {1 / np.asarray(list_duration_method_w_rendering).mean():.1f} fps")

    print(f"\nOutput video file: {outfn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Human mesh Recovery from a RGB video')
    parser.add_argument('--method', required=True, type=str, help='name of the method to use', choices=['mocapspin', 'mocapspin_posebert', 'dope_posebert'])
    parser.add_argument('--video', required=True, type=str, help='path to the video')
    parser.add_argument('--sideview', required=True, type=int, help='render sideview or not')
    args = parser.parse_args()
    main(args.video, args.method, args.sideview == 1)

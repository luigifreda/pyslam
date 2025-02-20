import torch
from lietorch import SO3, SE3, LieGroupParameter

import argparse
import numpy as np
import time
import torch.optim as optim
import torch.nn.functional as F


def draw(verticies):
    """ draw pose graph """
    import open3d as o3d

    n = len(verticies)
    points = np.array([x[1][:3] for x in verticies])
    lines = np.stack([np.arange(0,n-1), np.arange(1,n)], 1)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    o3d.visualization.draw_geometries([line_set])

def info2mat(info):
    mat = np.zeros((6,6))
    ix = 0
    for i in range(mat.shape[0]):
        mat[i,i:] = info[ix:ix+(6-i)]
        mat[i:,i] = info[ix:ix+(6-i)]
        ix += (6-i)

    return mat

def read_g2o(fn):
    verticies, edges = [], []
    with open(fn) as f:
        for line in f:
            line = line.split()
            if line[0] == 'VERTEX_SE3:QUAT':
                v = int(line[1])
                pose = np.array(line[2:], dtype=np.float32)
                verticies.append([v, pose])

            elif line[0] == 'EDGE_SE3:QUAT':
                u = int(line[1])
                v = int(line[2])
                pose = np.array(line[3:10], dtype=np.float32)
                info = np.array(line[10:], dtype=np.float32)

                info = info2mat(info)
                edges.append([u, v, pose, info, line])

    return verticies, edges

def write_g2o(pose_graph, fn):
    import csv
    verticies, edges = pose_graph
    with open(fn, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for (v, pose) in verticies:
            row = ['VERTEX_SE3:QUAT', v] + pose.tolist()
            writer.writerow(row)
        for edge in edges:
            writer.writerow(edge[-1])

def reshaping_fn(dE, b=1.5):
    """ Reshaping function from "Intrinsic consensus on SO(3), Tron et al."""
    ang = dE.log().norm(dim=-1)
    err = 1/b - (1/b + ang) * torch.exp(-b*ang)
    return err.sum()

def gradient_initializer(pose_graph, n_steps=500, lr_init=0.2):
    """ Riemannian Gradient Descent """

    verticies, edges = pose_graph

    # edge indicies (ii, jj)
    ii = np.array([x[0] for x in edges])
    jj = np.array([x[1] for x in edges])
    ii = torch.from_numpy(ii).cuda()
    jj = torch.from_numpy(jj).cuda()

    Eij = np.stack([x[2][3:] for x in edges])
    Eij = SO3(torch.from_numpy(Eij).float().cuda())

    R = np.stack([x[1][3:] for x in verticies])
    R = SO3(torch.from_numpy(R).float().cuda())
    R = LieGroupParameter(R)

    # use gradient descent with momentum
    optimizer = optim.SGD([R], lr=lr_init, momentum=0.5)

    start = time.time()
    for i in range(n_steps):
        optimizer.zero_grad()

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_init * .995**i

        # rotation error
        dE = (R[ii].inv() * R[jj]) * Eij.inv()
        loss = reshaping_fn(dE)

        loss.backward()
        optimizer.step()

        if i%25 == 0:
            print(i, lr_init * .995**i, loss.item())

    # convert rotations to pose3
    quats = R.group.data.detach().cpu().numpy()

    for i in range(len(verticies)):
        verticies[i][1][3:] = quats[i]

    return verticies, edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', help="input pose graph optimization file (.g2o format)")
    args = parser.parse_args()

    output_path = args.problem.replace('.g2o', '_rotavg.g2o')
    input_pose_graph = read_g2o(args.problem)

    rot_pose_graph = gradient_initializer(input_pose_graph)
    write_g2o(rot_pose_graph, output_path)


import time
import os
import torch
import torch.nn.functional as F
from psbody.mesh import Mesh
from utils.visualizer import Visualizer
from collections import OrderedDict

def run(model, train_loader, test_loader, args, optimizer, scheduler, writer,
        device, meshdata, mesh):
    train_losses, test_losses = [], []
    # load the checkpoint
    if writer.args.checkpoint is not None:
        print("Loading checkpoint %s" % (writer.args.checkpoint))
        writer.load_state_dict(model, optimizer, scheduler)
    
    visualizer = Visualizer(args)
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        train_loss = train(model, optimizer, train_loader, device, args, epoch, meshdata, mesh, visualizer)
        t_duration = time.time() - t
        test_loss = test(model, test_loader, device)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': args.epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration
        }

        writer.print_info(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch)


def train(model, optimizer, loader, device, args, epoch, meshdata, mesh, visualizer):
    model.train()
    total_loss = 0
    total_steps = 0
    mean = meshdata.mean
    std = meshdata.std
    for data in loader:
        num_graphs = data.num_graphs
        optimizer.zero_grad()
        x = data.x.to(device)
        out = model(x)
        loss = F.l1_loss(out, x, reduction='mean')
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        if total_steps % args.visualize_freq == 0:
            # TODO: visualize
            # TODO: move this to get_current_losses
            # print("inside visualize")
            losses = OrderedDict()
            losses["loss_l1"] = loss.item()
            losses["loss_l1_rep"] = loss.item()

            visualizer.plot_current_losses(epoch, float(total_steps) / len(loader), args, losses)

            # display result
            visual_results = {
                "input": (x.view(num_graphs, -1, 3).cpu() * std) + mean,
                "output": (out.view(num_graphs, -1, 3).cpu() * std) + mean,
                "faces": mesh.f
            }
            visualizer.display_current_results(visual_results, epoch, False)

        total_steps += 1

        

    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.x.to(device)
            pred = model(x)
            total_loss += F.l1_loss(pred, x, reduction='mean')
    return total_loss / len(loader)


def eval_error(model, test_loader, device, meshdata, args, mesh):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    # mesh = Mesh(filename=template_fp)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            pred = model(x)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean

            # reshaped_pred *= 1000
            # reshaped_x *= 1000
            # try to export the meshes
            mesh.v = (pred.view(-1, 3).cpu() * std) + mean
            mesh.write_obj(os.path.join(args.out_dir, "test", "output_%04d.obj" % i))

            mesh.v = (x.view(-1, 3).cpu() * std) + mean
            mesh.write_obj(os.path.join(args.out_dir, "test", "input_%04d.obj" % i))

            #mesh.show()

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_x)**2,
                          dim=2))  # [num_graphs, num_nodes]
            errors.append(tmp_error)
        new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]

        mean_error = new_errors.view((-1, )).mean()
        std_error = new_errors.view((-1, )).std()
        median_error = new_errors.view((-1, )).median()

    message = 'Error: {:.3f}+{:.3f} | {:.3f}'.format(mean_error, std_error,
                                                     median_error)

    out_error_fp = args.out_dir + '/euc_errors.txt'
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))
    print(message)

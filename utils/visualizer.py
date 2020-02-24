import numpy as np
import os
import sys
# import ntpath
import time
# from utils.utils import save_image
# from scipy.misc import imresize

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


# save image to the disk
# def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
#     image_dir = webpage.get_image_dir()
#     short_path = ntpath.basename(image_path[0])
#     name = os.path.splitext(short_path)[0]

#     webpage.add_header(name)
#     ims, txts, links = [], [], []

#     for label, im_data in visuals.items():
#         im = util.tensor2im(im_data)
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(image_dir, image_name)
#         h, w, _ = im.shape
#         if aspect_ratio > 1.0:
#             im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
#         if aspect_ratio < 1.0:
#             im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
#         util.save_image(im, save_path)

#         ims.append(image_name)
#         txts.append(label)
#         links.append(image_name)
#     webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    def __init__(self, opt):
        self.win_loss = 0
        self.win_A = 1
        self.win_B = 2
        self.opt = opt
        # self.env = opt.display_env
        self.name = opt.exp_name
        # self.Nw = opt.sliding_window
        # self.use_html = opt.isTrain and not opt.no_html
        # self.win_size = opt.display_winsize

        #  self.ncols = opt.display_ncols

        # if opt.use_display_freq:
        import visdom
        # self.vis = visdom.Visdom(
        #     server=opt.display_server, port=opt.display_port, env=opt.display_env, raise_exceptions=True)

        self.vis = visdom.Visdom(raise_exceptions=True)

        # if self.use_html:
        #     self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        #     self.img_dir = os.path.join(self.web_dir, 'images')
        #     print('create web directory %s...' % self.web_dir)
        #     util.mkdirs([self.web_dir, self.img_dir])

        # self.log_path = os.path.join(
        #     opt.logFolder, opt.name + '_training_loss_' + opt.timeStr + '.log')

        # with open(self.log_path, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write(
        #         '================ Training Loss (%s) ================\n' % now)

    # def reset(self):
    #     self.saved = False

    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        ''' for current batch, draw only the first pair of A, B
        '''
        # input_data = visuals['input_data']
        # A = input_data['A_cpu'][0]  # [9 * Nw, h, w] (-1 ,1)
        # B = input_data['B_cpu'][0]  # [3, h, w] (-1 ,1)
        # h, w = A.shape[1], A.shape[2]
        # fake_B = visuals['fake_B'][0]  # [3, h, w] (-1 ,1), on device
        # f2f_paths = []
        # pncc_paths = []
        # eye_paths = []
        # for i in range(self.Nw):
        #     f2f_paths.append(input_data['f2f_paths'][i][0])
        #     pncc_paths.append(input_data['pncc_paths'][i][0])
        #     eye_paths.append(input_data['eye_paths'][i][0])

        # # f2f_paths = input_data['f2f_paths']  # list , Nw,
        # # pncc_paths = input_data['pncc_paths']  # list , Nw,
        # # eye_paths = input_data['eye_paths']  # list , Nw,
        # image_path = input_data['img_path'][0]  # str

        # # torch -> (float) numpy -> (0, 1)
        # A_np = (A.cpu().float().numpy() + 1) * 0.5
        # B_np = (B.cpu().float().numpy() + 1) * 0.5
        # fake_B_np = (fake_B.detach().cpu().float().numpy() + 1) * 0.5

        # # f2f_imgs = A_np[0: 3 * self.Nw].reshape(self.Nw, 3, -1, -1)  # [3 * Nw, h, w]
        # # # [3 * Nw, h, w]
        # # pncc_imgs = A_np[3 * self.Nw: 6 * self.Nw].reshape(self.Nw, 3, -1, -1)
        # # # [3 * Nw, h, w]
        # # eye_imgs = A_np[6 * self.Nw: 9 * self.Nw].reshape(self.Nw, 3, -1, -1)

        # f2f_caption = ""
        # pncc_caption = ""
        # eye_caption = ""
        # for i in range(self.Nw):
        #     f2f_caption += os.path.basename(f2f_paths[i]) + " "
        #     pncc_caption += os.path.basename(pncc_paths[i]) + " "
        #     eye_caption += os.path.basename(eye_paths[i]) + " "
        input_v = visuals["input"][0]
        output_v = visuals["output"][0]
        faces = visuals["faces"]
        try:
            # self.vis.images(A_np.reshape((self.Nw * 3, 3, h, w)),
            #                 nrow=self.Nw, win=self.win_A, opts=dict(title='Input'))

            self.vis.mesh(X=input_v, Y=faces, win=self.win_A, opts=dict(title='Input', opacity=1.0))
            self.vis.mesh(X=output_v, Y=faces, win=self.win_B, opts=dict(title='Output', opacity=1.0))

            # self.vis.images([B_np, fake_B_np],
            #                 nrow=self.Nw, win=self.win_B, opts=dict(title='Output'))

        except VisdomExceptionBase:
            self.throw_visdom_connection_error()

        # if self.display_id > 0:  # show images in the browser
        #     ncols = self.ncols
        #     if ncols > 0:
        #         ncols = min(ncols, len(visuals))
        #         h, w = next(iter(visuals.values())).shape[:2]
        #         table_css = """<style>
        #                 table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
        #                 table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
        #                 </style>""" % (w, h)
        #         title = self.name
        #         label_html = ''
        #         label_html_row = ''
        #         images = []
        #         idx = 0
        #         for label, image in visuals.items():
        #             image_numpy = util.tensor2im(image)
        #             label_html_row += '<td>%s</td>' % label
        #             images.append(image_numpy.transpose([2, 0, 1]))
        #             idx += 1
        #             if idx % ncols == 0:
        #                 label_html += '<tr>%s</tr>' % label_html_row
        #                 label_html_row = ''
        #         white_image = np.ones_like(
        #             image_numpy.transpose([2, 0, 1])) * 255
        #         while idx % ncols != 0:
        #             images.append(white_image)
        #             label_html_row += '<td></td>'
        #             idx += 1
        #         if label_html_row != '':
        #             label_html += '<tr>%s</tr>' % label_html_row
        #         # pane col = image row
        #         try:
        #             self.vis.images(images, nrow=ncols, win=self.display_id + 1,
        #                             padding=2, opts=dict(title=title + ' images'))
        #             label_html = '<table>%s</table>' % label_html
        #             self.vis.text(table_css + label_html, win=self.display_id + 2,
        #                           opts=dict(title=title + ' labels'))
        #         except VisdomExceptionBase:
        #             self.throw_visdom_connection_error()

        #     else:
        #         idx = 1
        #         for label, image in visuals.items():
        #             image_numpy = util.tensor2im(image)
        #             self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
        #                            win=self.display_id + idx)
        #             idx += 1

        # save images to a html file
        if save_result:
            print("pass in save result")
            pass
            # fake_B_out = (np.transpose(fake_B_np, (1, 2, 0))
            #               * 255).astype(np.uint8)
            # img_path = os.path.join(
            #     self.opt.outputImgFolder, 'epoch%.3d_%s' % (epoch, os.path.basename(image_path)))
            #save_image(fake_B_out, img_path)
            # update website
            # webpage = html.HTML(
            #     self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            # for n in range(epoch, 0, -1):
            #     webpage.add_header('epoch [%d]' % n)
            #     ims, txts, links = [], [], []

            #     for label, image_numpy in visuals.items():
            #         image_numpy = util.tensor2im(image)
            #         img_path = 'epoch%.3d_%s.png' % (n, label)
            #         ims.append(img_path)
            #         txts.append(label)
            #         links.append(img_path)
            #     webpage.add_images(ims, txts, links, width=self.win_size)
            # webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k]
                                    for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] *
                           len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.win_loss)
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()

    # # losses: same format as |losses| of plot_current_losses
    # def print_current_losses(self, epoch, i, losses, t, t_data):
    #     message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (
    #         epoch, i, t, t_data)
    #     for k, v in losses.items():
    #         message += '%s: %.3f ' % (k, v)

    #     print(message)
    #     with open(self.log_path, "a") as log_file:
    #         log_file.write('%s\n' % message)

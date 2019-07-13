
'''
kitti object label
'''
import os
import numpy as np
import cv2
import argparse
from xml.dom.minidom import Document

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--tag', type=str, nargs='?', default='000000', help='set input data filename')

args = parser.parse_args()

kitti_dir = './kitti/'

class KittiObject:

#what about detection result??? 

    def __init__(self, label_str):

        data = label_str.strip().split()
        self.type_name = data[0]
        #---how about detection result?
        self.truncated = data[1]
        self.occluded = data[2]
        self.alpha = np.float(data[3])
        #---we may not need these three properties

        #(left, top, right, bottom)
        self.xmin = np.float(data[4])
        self.ymin = np.float(data[5])
        self.xmax = np.float(data[6])
        self.ymax = np.float(data[7])
        self.bbox2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        
        #(x,y,z,h,w,l,rot_y)
        self.t = (np.float(data[11]), np.float(data[12]), np.float(data[13]))
        self.h = np.float(data[8])
        self.w = np.float(data[9])
        self.l = np.float(data[10])
        self.ry = np.float(data[-1]) 
        #self.bbox3d_center = np.array([self.t[0], self.t[1], self.t[2], self.h, self.w, self.l) 

    def get_bbox3d_corner(self):
        #argument: bbox3d_center (x, y, z, h, w, l, rot_y)     h,w,l  x<->l, y<->h, z<->w
        #return: bbox3d_corner shape: 8, 3: 8 corner points, 3 axises
        #default corners * rotation + translation(center point)
    
        '''
        different orientation correspond different bbox
        0 starts from left side of the direction the car is driving on, right side is 1..and so on..
        give a example when car is on the left side of camera and heading to camera
         6  7
        2  3
         5  4
        1  0
        '''
   
        translation = np.array(self.t) 
        h, w, l = self.h, self.w, self.l
        ry = self.ry
        rotation = np.array([
                        [np.cos(ry), 0.0, np.sin(ry)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(ry), 0.0, np.cos(ry)]])

        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        bbox3d_corner = np.dot(rotation, np.vstack([x_corners, y_corners, z_corners]))
        bbox3d_corner += translation.reshape((3,1))
        #shape (3,8)
        bbox3d_corner = bbox3d_corner.transpose()
        #shape (8, 3)
        #add midpoint

        return bbox3d_corner


       
    def get_nearest_camera(self):
        #birdview, 16 viewpoints = 8 birdview(4 corner + 4 middle) * 2 frontview(up + down)
        #take the nearest one as the viewpoint 
        #bbox3d_center:(x,y,z,h,w,l)
    
        bbox3d_corner = self.get_bbox3d_corner()
        bbox3d_mid = np.zeros((8, 3))
        for k in range(4):
            i, j = k, (k+1)%4
            bbox3d_mid[i] = (bbox3d_corner[i]+bbox3d_corner[j]) / 2
            i, j = k+4, (k+1)%4
            bbox3d_mid[i] = (bbox3d_corner[i]+bbox3d_corner[j]) / 2
            bbox3d_total = np.vstack((bbox3d_corner, bbox3d_mid))
        # point_idx [down_corner, up_corner, down_mid, up_mid]
        distance = np.sum(np.square(bbox3d_total), axis=1)
        nearest_idx = np.argmin(distance)  
        #another_nearest = (nearest_idx % 8 + 4 ) % 8 + (nearest_idx / 8) * 8
        #nearest_mid = (bbox3d_total[nearest_idx] + bbox3d_total[another_nearest]) / 2
        #distance_mid = np.sum(np.square(nearest_mid))
        
        distance_center = np.sqrt(np.sum(np.square(self.t)))
        y_angle = np.arccos(self.t[1] / distance_center)
        #all 80+ ??????
        print('angle between y-axis and obj center to camera center {}'.format(y_angle * 180 / np.pi))        
        #whats the difference between front view and top  view.....
        if nearest_idx % 8 >= 4:
            #difference = distance_mid - distance[nearest_idx] 
            #print('difference dist between nearest {} and mid  is {}'.format(nearest_idx, difference))
            if y_angle >  85: #difference < 0.5:
                print('camera should be looking front..') 
            else:
                print('camera looking down... too near from camera') 
        else:
            print('camera looking up(flat view) ... too far from camera')

        #if the nearest is down_corner, then camera is looking up..  
        vertical_view = ['front', 'top']
        vertical_idx = np.int(nearest_idx % 8 / 4)
        print(vertical_idx)


        horizontal_view = ['left head', 'right head', 'right tail', 'left tail', \
                           'head mid', 'right mid', 'tail mid', 'left mid' ]
        horizontal_idx = np.int((nearest_idx % 8) % 4+ (nearest_idx / 8) * 4)
        print('nearest idx ', nearest_idx)
        print('view from nearest point vertical: {} horizontal: {}' \
                .format(vertical_view[vertical_idx], horizontal_view[horizontal_idx]))

 
        return nearest_idx

    
 
def draw_bbox3d_on_image(img, bbox3d_corner, P, color, viewpoint=-1):
    #bbox3d_center should transformed to bbox3d_corner
    # P is the projection matrix projecting 3d point to 2d image plane

    #homogenous coordinate
    bbox3d_corner_homo = np.hstack((bbox3d_corner, np.ones((8, 1))))
    
    bbox3d_img = np.dot(P, bbox3d_corner_homo.transpose())
    bbox3d_img /= bbox3d_img[2, :]
    bbox3d_img = bbox3d_img[:2]

    thickness = 1 
    bbox3d_img = bbox3d_img.astype(np.int).transpose()
    for k in range(4):
        #(0,1,2,3) 
        i, j = k, (k+1)%4
        cv2.line(img, (bbox3d_img[i, 0], bbox3d_img[i, 1]), (bbox3d_img[j, 0], bbox3d_img[j, 1]), color, thickness)
        #(4,5,6,7)
        i, j = k+4, (k+1)%4 + 4
        cv2.line(img, (bbox3d_img[i, 0], bbox3d_img[i, 1]), (bbox3d_img[j, 0], bbox3d_img[j, 1]), color, thickness)
        i, j = k, k+4
        cv2.line(img, (bbox3d_img[i, 0], bbox3d_img[i, 1]), (bbox3d_img[j, 0], bbox3d_img[j, 1]), color, thickness) 
    
        
    #view = ['tail mid', 'left tail', 'left mid', 'left head', 'head mid', 'right head', 'right mid', 'right tail']
    #view_idx = (viewpoint - (-1*np.pi)) / (2*np.pi/8)
    
    if viewpoint >= 0:   
        #print_bbox3d_viewpoint(viewpoint)
        cv2.circle(img, (bbox3d_img[viewpoint % 8, 0], bbox3d_img[viewpoint % 8, 1]), 5, [255,255,125], 3)


def draw_bbox2d_on_image(img, bbox2d, color):
   
    #bbox2d: shape (4,) left, top, right, bottom)
    bbox2d = bbox2d.astype(int)
    thickness = 1
    cv2.line(img, (bbox2d[0], bbox2d[1]), (bbox2d[0], bbox2d[3]), color, thickness, cv2.LINE_AA)
    cv2.line(img, (bbox2d[0], bbox2d[1]), (bbox2d[2], bbox2d[1]), color, thickness, cv2.LINE_AA)
    cv2.line(img, (bbox2d[2], bbox2d[3]), (bbox2d[2], bbox2d[1]), color, thickness, cv2.LINE_AA)
    cv2.line(img, (bbox2d[2], bbox2d[3]), (bbox2d[0], bbox2d[3]), color, thickness, cv2.LINE_AA)


def visualize_object():
    with open(os.path.join(kitti_dir, 'calib', 'calib_cam_to_cam.txt'), 'r') as f_calib:
        for line in f_calib.readlines():
            if line.startswith('P_rect_02'):
                p_rect_02 = line.strip().split(':')[1]
                break

    P = np.fromstring(p_rect_02, sep=' ')
    P = P.reshape((3, 4))
    print('project matrix\n {}'.format(P))
     
    #read label
    tag = args.tag #'000055'
    img = cv2.imread(os.path.join(kitti_dir, tag+'.png'))
    color = (0,255,255)
    gt_color = (255, 0, 255)
 
    with open(os.path.join(kitti_dir, tag+'.txt'),'r') as f_label:    
        for object_idx, line in enumerate(f_label.readlines()):
            #from string to array!
            if line.startswith('DontCare'):
                continue
            print(line)
            obj = KittiObject(line)
            draw_bbox2d_on_image(img, obj.bbox2d, gt_color)
            viewpoint = obj.get_nearest_camera()
            draw_bbox3d_on_image(img, obj.get_bbox3d_corner(), P, color, viewpoint)
            cv2.imshow(tag+'.png', img) 
            cv2.waitKey(0)
 
    cv2.destroyAllWindows()


def convert_to_xml(source_dir, target_dir, filename):
    f_source = os.path.join(source_dir, 'label_2', filename+'.txt')
    f_image = os.path.join(source_dir, 'image_2', filename+'.png')
    f_target = os.path.join(target_dir, filename+'.xml')
    #print(f_source, f_image, f_target)
    doc = Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('KITTI')
    title.appendChild(title_text)
    annotation.appendChild(title)

    img_size = cv2.imread(f_image).shape

    title = doc.createElement('filename')
    title_text = doc.createTextNode(filename+'.png')
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The KITTI Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('KITTI')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    with open(f_source, 'r') as f:
        for line in f.readlines():
            if line.startswith('DontCare') or line.startswith('Misc'):
                continue
            obj = KittiObject(line) 
            if obj.type_name in ['Van', 'Truck', 'Tram']:
                obj.type_name = 'Car'
            elif obj.type_name in ['Person_sitting']:
                obj.type_name = 'Pedestrian'
            obj_node = doc.createElement('object')
            annotation.appendChild(obj_node)

            title = doc.createElement('name')
            title_text = doc.createTextNode(obj.type_name)
            title.appendChild(title_text)
            obj_node.appendChild(title)

            bndbox = doc.createElement('bndbox')
            obj_node.appendChild(bndbox)
            title = doc.createElement('xmin')
            title_text = doc.createTextNode(str(int(obj.xmin)))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymin')
            title_text = doc.createTextNode(str(int(obj.ymin)))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('xmax')
            title_text = doc.createTextNode(str(int(obj.xmax)))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymax')
            title_text = doc.createTextNode(str(int(obj.ymax)))
            title.appendChild(title_text)
            bndbox.appendChild(title)


    with open(f_target, 'w') as f:
        f.write(doc.toprettyxml(indent=''))
   

def convert(): 
   
    split_file = '/home/guest/yeungly/kitti/split_files/train.txt' 
    kitti_dir = '/home/guest/yeungly/kitti/training' 
    with open(split_file, 'r') as f_tag:
        for tag in f_tag.readlines()[:10]:
            tag = tag.strip()
            print('converting {}...'.format(tag))
            convert_to_xml(kitti_dir, kitti_dir+'/annotations', tag)

    #convert_to_xml(kitti_dir, kitti_dir+'/annotations', args.tag)


if __name__ == '__main__':

    visualize_object()


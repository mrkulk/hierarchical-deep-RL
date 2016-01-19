import zmq
import time
import sys
import cv2
import numpy as np
import copy
import sys
import json, pdb

port = "5550"
if len(sys.argv) > 1:
    port =  int(sys.argv[1])

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)


class Recognizer:
	def __init__(self):
		self.colors = {'man': [200, 72, 72], 'skull': [236,236,236]}
		self.map = {'man': 0, 'skull': 1, 'ladder': 2, 'door': 3, 'key': 4}

	def blob_detect(self, img, id):
		mask = np.zeros(np.shape(img))
		mask[:,:,0] = self.colors[id][0];
		mask[:,:,1] = self.colors[id][1];
		mask[:,:,2] = self.colors[id][2];

		diff = img - mask
		indxs = np.where(diff == 0)
		diff[np.where(diff < 0)] = 0
		diff[np.where(diff > 0)] = 0
		diff[indxs] = 255
		mean_y = np.sum(indxs[0]) / np.shape(indxs[0])[0]
		mean_x = np.sum(indxs[1]) / np.shape(indxs[1])[0]
		return (mean_y, mean_x) #flipped co-ords due to numpy blob detect
		# return (mean_x, mean_y)

	def template_detect(self, img, id):
		template = cv2.imread('templates/' + id + '.png')
		w = np.shape(template)[1]
		h = np.shape(template)[0]
		res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
		threshold = 0.8
		loc = np.where( res >= threshold)
                loc[0].setflags(write=True)
                loc[1].setflags(write=True)
		for i in range(np.shape(loc[0])[0]):
			loc[0][i] += h/2; loc[1][i] += w/2
		return loc, w, h

	def get(self, img):
		#detect man
		man_coords = self.blob_detect(img, 'man')
		skull_coords = self.blob_detect(img, 'skull')
		ladder_coords, ladder_w, ladder_h = self.template_detect(img, 'ladder')
		key_coords, key_w, key_h = self.template_detect(img, 'key')
		door_coords, door_w, door_h = self.template_detect(img, 'door_new')
		return {'man': man_coords, 'skull':skull_coords, 'ladder':ladder_coords, 'key':key_coords, 'door':door_coords, 'ladder_w': ladder_w,
		'ladder_h':ladder_h	, 'key_w':key_w, 'key_h':key_h, 'door_w':door_w, 'door_h':door_h}

	def drawbbox(self, inputim, coords):
		img = copy.deepcopy(inputim)
		for id in {'ladder', 'key', 'door'}:
			for pt in zip(*coords[id][::-1]):
			  cv2.rectangle(img, pt, (pt[0] + coords[id+'_w'], pt[1] + coords[id+'_h']), (0,0,255), 2)
		cv2.rectangle(img, (coords['man'][0] - 5, coords['man'][1] - 5), (coords['man'][0] + 5, coords['man'][1] + 5),  (0,0,255), 2)
		cv2.rectangle(img, (coords['skull'][0] - 5, coords['skull'][1] - 5), (coords['skull'][0] + 5, coords['skull'][1] + 5),  (0,0,255), 2)
		return img

	def get_lives(self, img):
		return np.sum(img)

	def get_onehot(self, ID):
		tmp = list(np.zeros(len(self.map)))
		tmp[ID] = 1
		return tmp

	def process_objects(self, objects):
		objects_list = []

		objects_list.append([objects['man'][0], objects['man'][1]] + self.get_onehot(self.map['man']))
		objects_list.append([objects['skull'][0], objects['skull'][1]] + self.get_onehot(self.map['skull']))
		
		for obj, val in objects.items():
			# print(obj, val)
			if obj is not 'man' and obj is not 'skull':
				if type(val) is not type(1):
					if type(val[0]) == np.int64:
						objects_list.append([val[0], val[1]]  + self.get_onehot(self.map[obj]))
					else:
						for i in range(np.shape(val[0])[0]):
							objects_list.append([val[0][i], val[1][i]] + self.get_onehot(self.map[obj]))
		#process objects and pad with zeros to ensure fixed length state dim
		fill_objects = 8 - len(objects_list)
		for j in range(fill_objects):
			objects_list.append([0, 0] + list(np.zeros(len(self.map))))

		return objects_list


def show(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	# cv2.destroyAllWindows()

def unit_test():
    rec = Recognizer()
    try:
        img_id = str(sys.argv[1])
    except:
        print 'Using default image 1.png'
        img_id = '1'
    img_rgb = cv2.imread('tmp.png')
    im_score = img_rgb[15:20, 55:95, :]
    img_rgb = img_rgb[30:,:,:]
    coords = rec.get(img_rgb)
    objects = rec.process_objects(coords)
    pdb.set_trace()
    img = rec.drawbbox(img_rgb, coords)
    show(img)

# unit_test()

rec = Recognizer()

img_rgb = cv2.imread('tmp.png')
im_score = img_rgb[15:20, 55:95, :]
img_rgb = img_rgb[30:,:,:]
coords = rec.get(img_rgb)
objects_list_cache = rec.process_objects(coords)

while True:
    #  Wait for next request from client
    message = socket.recv()
    # print "Received request: ", message
    img_rgb = cv2.imread('tmp_'+str(port)+'.png')
    im_score = img_rgb[15:20, 55:95, :]
    img_rgb = img_rgb[30:,:,:]
    coords = rec.get(img_rgb)
    # img = rec.drawbbox(img_rgb, coords)
    # show(img)  
    objects_list = copy.deepcopy(objects_list_cache)
    objects_list2 = rec.process_objects(coords)
    #agent and skull is dynamic. everything else is static. TODO for key
    objects_list[0] = objects_list2[0] 
    objects_list[1] = objects_list2[1]
    if objects_list[1][0] == 0 and objects_list[1][1] == 0:
    	objects_list[1][3] = 0
    # print(len(objects_list))
    socket.send('objlist = '+json.dumps(objects_list).replace('[','{').replace(']','}'))
    # socket.send("World from %s" % str(coords))
    # print(rec.get_lives(im_score))


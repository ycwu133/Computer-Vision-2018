import os
import argparse
import numpy as np
import cv2
import math
inf = math.inf

# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=3, help='Top N most voted candidates for each input image...')
parser.add_argument("--image", default="0a.png", help='Image for rgb2gray conversion...')
parser.add_argument("--input", default='./testdata/', help = "Testing data path..." )
args = parser.parse_args()



# set weight space matrix 
c1 = np.array([[0,0,0,0,0,0,0,0,0,0,0], 
			   [1,1,1,1,1,1,1,1,1,1,0], 
			   [2,2,2,2,2,2,2,2,2,0,0], 
			   [3,3,3,3,3,3,3,3,0,0,0], 
			   [4,4,4,4,4,4,4,0,0,0,0], 
			   [5,5,5,5,5,5,0,0,0,0,0],
	           [6,6,6,6,6,0,0,0,0,0,0], 
	           [7,7,7,7,0,0,0,0,0,0,0], 
	           [8,8,8,0,0,0,0,0,0,0,0], 
	           [9,9,0,0,0,0,0,0,0,0,0], 
	          [10,0,0,0,0,0,0,0,0,0,0]])

c2 = c1.T
#c3 = 10 - c1 -c2



def weight_space_preprocessing(image, c1, c2, c3):
	# Quantize the weight space , RGB cofficients : (c1, c2, c3)
	
	(B,G,R) = cv2.split(image)

	image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

	image_yuv[:, :, 0] = c1*R + c2*G + c3*B # y channel

	image_gray = image_yuv[:, :, 0]

	return image_gray


def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))


def joint_bilateral_filter(source, guidance, sigma_r, sigma_s):
	# Joint bilateral filter for one channel
    
	# set filter radius
    radius = 3 * sigma_s

    # image normalization
    source = source.astype("float") / 255.0
    guidance = guidance.astype("float") / 255.0

    filtered_image = np.zeros_like(source).astype(float)
    W = 0

    pad_guidance = np.pad(guidance, (radius, radius), mode="symmetric")
    pad_src = np.pad(source, (radius, radius), mode="symmetric")
    
    for i in range(-radius, radius+1):  
        for j in range(-radius, radius+1):
 
            window_guidance = pad_guidance[radius + i: radius + i + guidance.shape[0], radius + j: radius + j + guidance.shape[1]]
            window_src = pad_src[radius + i: radius + i + source.shape[0], radius + j: radius + j + source.shape[1]]

            distance_x = (i)**2
            distance_y = (j)**2
            distance = np.sqrt(distance_x + distance_y)

            gi = gaussian((window_guidance - guidance), sigma_r)
            gs = gaussian(distance, sigma_s)

            W += gi * gs
            filtered_image +=  window_src * ( gi * gs ) * 255

    return np.around(filtered_image / W).astype(np.uint8)


def joint_bilateral_filter_color(source, guidance, sigma_r, sigma_s):

    (b,g,r) = cv2.split(image)  

    Bout=joint_bilateral_filter(b, guidance, sigma_r, sigma_s)
    Gout=joint_bilateral_filter(g, guidance, sigma_r, sigma_s)
    Rout=joint_bilateral_filter(r, guidance, sigma_r, sigma_s)

    merged = cv2.merge([Bout,Gout,Rout])

    return merged

def joint_bilateral_filter_color_reference(source, sigma_r, sigma_s):

    (b,g,r) = cv2.split(image)  

    Bout=joint_bilateral_filter(b, b, sigma_r, sigma_s)
    Gout=joint_bilateral_filter(g, g, sigma_r, sigma_s)
    Rout=joint_bilateral_filter(r, r, sigma_r, sigma_s)

    merged = cv2.merge([Bout,Gout,Rout])

    return merged

def cost_function(image, image_edited):

	diff = cv2.subtract(image, image_edited)

	cost = abs(np.sum(diff))
	
	return cost


def local_min(cost_m):
	
	vote_m = np.zeros(shape=(11,11))

	for k in range(11):
		for l in range(11):		
			
			if k < 10 and l == 0:	
				if cost_m[k,l] < cost_m[k+1,l] and cost_m[k,l] < cost_m[k-1,l] and cost_m[k,l] < cost_m[k+1,l+1] and cost_m[k,l] < cost_m[k-1,l+1] and cost_m[k,l] < cost_m[k,l+1]: 
					vote_m[k,l] = 1		
	
			if k == 0 and l < 10:	
				if cost_m[k,l] < cost_m[k,l-1] and cost_m[k,l] < cost_m[k,l+1] and cost_m[k,l] < cost_m[k+1,l-1] and cost_m[k,l] < cost_m[k+1,l+1] and cost_m[k,l] < cost_m[k+1,l]: 
					vote_m[k,l] = 1		
							
			if 0< k < 10 and 0< l < 10:	
				if cost_m[k,l] < cost_m[k+1,l] and cost_m[k,l] < cost_m[k-1,l] and cost_m[k,l] < cost_m[k,l+1] and cost_m[k,l] < cost_m[k,l-1] and cost_m[k,l] < cost_m[k+1,l+1] and cost_m[k,l] < cost_m[k-1,l+1] and cost_m[k,l] < cost_m[k-1,l-1] and cost_m[k,l] < cost_m[k+1,l-1]: 
					vote_m[k,l] = 1
					
			if cost_m[0,0] < cost_m[1,0] and cost_m[0,0] < cost_m[0,1] and cost_m[0,0] < cost_m[1,1]:
				vote_m[0,0] = 1
							
			if cost_m[0,10] < cost_m[1,10] and cost_m[0,10] < cost_m[1,9] and cost_m[0,10] < cost_m[0,9]:
				vote_m[0,10] = 1
					
			if cost_m[10,0] < cost_m[9,0] and cost_m[10,0] < cost_m[9,1] and cost_m[10,0] < cost_m[10,1]:
				vote_m[10,0] = 1	
	
	return(vote_m)

def voted_results(vote_result):	
	# calculating the voted numbers for each input image
	
	raw, column = vote_result.shape			
	
	position = np.argmax(vote_result)		
	max_value = np.max(vote_result)			
	m, n = divmod(position, column)
	
	print ("Voted Numbers : " ,max_value, "and the Weight for R is : ", m / 10, ", G is :", n /10 ,", B is : ", (10 - m - n) / 10 )
	print("\n")		
	
	vote_result[m,n] = 0

	return m / 10, n /10, (10 - m - n) / 10 


if __name__ == '__main__':
	
	image_path = os.path.join(args.input, args.image)
	image = cv2.imread(image_path)
	
	# Conventional rgb2gray conversion
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imwrite((os.path.splitext(args.image )[0] + "_y" + ".png"), gray_image)

	# Advanced rgb2gray conversion
	sigma_s = [1, 2, 3]
	sigma_r = [0.05, 0.1, 0.2]

	cost_matrix = np.zeros(shape=(11,11))
	vote_result = np.zeros(shape=(11,11))

	for r in sigma_r:
		for s in sigma_s:
			
			image_refference = joint_bilateral_filter_color_reference(image, r, s)
			
			for i in range(11):
				for j in range(11):

					image_gray = weight_space_preprocessing(image, c1[i,j] / 10, c2[i,j] / 10, (10 - c1[i,j] - c2[i,j]) / 10)

					image_jbf = joint_bilateral_filter_color(image, image_gray, r, s)

					cost = cost_function(image_refference, image_jbf)
					
					print('Cost : ', cost)
					print('Now The Weight, Wr : ', c1[i,j] / 10 , 'Wg : ', c2[i,j] / 10 , 'Wb : ' , (10 - c1[i,j] - c2[i,j]) / 10)
					print('Now The Sigma , Sigma_r :', r , 'Sigma_s : ', s)
					print()

					if i == 0 and j == 0 :
						cost_matrix[i,j] = cost

					elif c1[i,j] == 0 and c2[i,j] == 0 :
						cost_matrix[i,j] = inf

					else:
						cost_matrix[i,j] = cost	
			
			#print(cost_matrix)
			vote_result = vote_result + local_min(cost_matrix)
	
	#print(vote_result)
		
	for i in range(args.num):
		print("================================================================================")
		print("Top ", i+1)
		co1, co2, co3 = voted_results(vote_result)
		
		image_candidate = weight_space_preprocessing(image, co1, co2, co3)
		
		cv2.imshow("Display window", image_candidate) 
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		cv2.imwrite((os.path.splitext(args.image )[0] + "_y" + str(i+1)+ ".png"), image_candidate)
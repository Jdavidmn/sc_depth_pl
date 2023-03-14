import cv2
import configargparse

def get_opts():
	parser = configargparse.ArgumentParser()

	parser.add_argument('--input_file', type=str)
	parser.add_argument('--output_dir', type=str)
	parser.add_argument('--file_name', type=str, default='test.jpg')

	return parser.parse_args()

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):

	# Extraemos las dimensiones originales
	(original_height, original_width) = image.shape[:2]

	# Si el nuevo ancho es vacío, calculamos la relación de aspecto con base a la nueva altura
	if width is None:
		# Proporción para mantener la realción de aspecto con base a la nueva altura
		ratio = height / float(original_height)

		# Nueva anchura
		width = int(original_width * ratio)
	else:
		# Proporción para mantener la realción de aspecto con base a la nueva anchura
		ratio = width / float(original_width)

		# Nueva altura
		height = int(original_height * ratio)

	newsize = (width, height)

	# El nuevo tamaño de la imagen no será más que un para compuesto por la nueva anchura y la nueva altura
	# Usamos lafunción cv2.resize() para llevar a cabo el cambio de tamaño de la imagen
	# finalmente retornamos el resultado
	return cv2.resize(image, newsize, interpolation=inter)

def main():
	param = get_opts()
	image = cv2.imread(param.input_file)

#	cv2.imshow("original",image)

	(original_height, original_width) = image.shape[:2]

	new_height = int(original_height/original_width*640)
	new_width = int(original_width/original_height*384)

	if new_height < 384:
		resized = resize(image,height=384)
		resized = resized[0:384, (resized.shape[1]//2 - 320):(resized.shape[1]//2 + 320)]


	elif new_width < 640:
		resized = resize(image,width=640)
		resized = resized[(resized.shape[0]//2 - 192):(resized.shape[0]//2 + 192), 0:640]

#	resized = resize(image,height=384)

#	if resized.shape[1] > 640:
#		resized = resized[0:resized.shape[0], (resized.shape[1]//2 - 320):(resized.shape[1]//2 + 320)]
#	else:
#		delta_w = image.shape[1] - resized.shape[1]
#		left, right = delta_w//2, delta_w-(delta_w//2)
#		resized = cv2.copyMakeBorder(resized, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

#	cv2.imshow("resized", resized)

	cv2.imwrite(param.output_dir+'/'+param.file_name, resized)

#	cv2.waitKey(0)
#	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()

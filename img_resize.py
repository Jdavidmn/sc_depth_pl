import cv2


def img_resize(image, width=None, height=None, inter=cv2.INTER_AREA):

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

def resize(image):

	(original_height, original_width) = image.shape[:2]

	new_height = int(original_height/original_width*640)
	new_width = int(original_width/original_height*384)

	if new_height < 384:
		resized = img_resize(image,height=384)
		resized = resized[0:384, (resized.shape[1]//2 - 320):(resized.shape[1]//2 + 320)]


	elif new_width < 640:
		resized = img_resize(image,width=640)
		resized = resized[(resized.shape[0]//2 - 192):(resized.shape[0]//2 + 192), 0:640]

	return resized

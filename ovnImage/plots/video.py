import cv2


def images2video(images, video_name, fps):
    """
    Write a sequence of images as a video

    :param images: List of images
    :param video_name: String Name of the result video file
    :param fps: int Number of frames per second
    :return:
    """
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps=fps, frameSize=(width, height))
    for image in images:
        video.write(cv2.imread(image))

    video.release()

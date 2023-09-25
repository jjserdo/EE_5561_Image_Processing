September 24, 2023

#work
[[inpainting_exemplar.pdf]]

---

![[Pasted image 20230924150445.png]]

implementation is only up to page 6

def inspectImage():
    def inspect(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("image[{}, {}] = {}".format(y, x, raw_image[y, x, :] ))
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', inspect)
    
    while(True):
        cv2.imshow('image', raw_image)
        if cv2.waitKey(20) & 0xFF == 27: # ASCII character = 27 which is 'escape'
            break
    cv2.destroyAllWindows()


def inspectImage():
    def inspect(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("image[{}, {}] = {}".format(y, x, raw_image[y, x, :] ))
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', inspect)
    
    while(True):
        cv2.imshow('image', raw_image)
        if cv2.waitKey(20) & 0xFF == 27: # ASCII character = 27 which is 'escape'
            break
    cv2.destroyAllWindows()

[OpenCV Python – How to draw a rectangle using Mouse Events? (tutorialspoint.com)](https://www.tutorialspoint.com/opencv-python-how-to-draw-a-rectangle-using-mouse-events#:~:text=Steps%201%20Import%20required%20library%20OpenCV.%20Make%20sure,the%20image%20window%20%22%20Rectangle%20Window%20%22.%20)

[opencv - Drawing filled polygon using mouse events in open cv using python - Stack Overflow](https://stackoverflow.com/questions/37099262/drawing-filled-polygon-using-mouse-events-in-open-cv-using-python)

[A qualitative study of Exemplar based Image Inpainting | SN Applied Sciences (springer.com)](https://link.springer.com/article/10.1007/s42452-019-1775-7)
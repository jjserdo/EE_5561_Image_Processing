September 24, 2023

#work
[[inpainting_exemplar.pdf]]

---

For this mini-project, we will consider on image inpainting (see attached paper, [inpainting_exemplar.pdf](https://canvas.umn.edu/courses/391981/files/38027462?wrap=1 "Link") [Download inpainting_exemplar.pdf](https://canvas.umn.edu/courses/391981/files/38027462/download?download_frd=1)[Download inpainting_exemplar.pdf](https://canvas.umn.edu/courses/300316/files/26585584/download?download_frd=1) ). The mini-project involves the following:

1. Read & understand this paper
2. Implementing the processing techniques proposed in the work in Python (or similar software), using either numerical or real-life datasets.
3. Submit a written report in IEEE conference format, along with the Python source files to re-generate the results. The report should include an abstract, and the following sections: Introduction, Methods, Results, and Discussion; should be approximately 4-pages in two-column format, excluding references. The Discussion section should identify the limitations of the technique (regardless of the success of your own Python implementation).

This first mini-project is due on October 19 (11:59 pm).

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


COnverting RGB to Lab
[OpenCV Color Spaces ( cv2.cvtColor ) - PyImageSearch](https://pyimagesearch.com/2021/04/28/opencv-color-spaces-cv2-cvtcolor/)
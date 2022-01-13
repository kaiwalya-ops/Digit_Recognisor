import numpy as np
from keras.models import load_model
from numpy.lib.type_check import imag
import pygame,sys
from pygame import font
from pygame import display
from pygame.draw import circle
from pygame.font import Font
from pygame.locals import *
import cv2

PREDICT=True
IMAGESAVE=False
imgcnt=0
SIZEX,SIZEY=640,480
(WHITE,BLACK,RED)=((255,255,255),(0,0,0),(255,0,0))
MODEL= load_model('D:\\Programs\\Digit_Recognisor\\model')
LABELS = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']

pygame.init()
DISPLAY = pygame.display.set_mode((SIZEX,SIZEY))
pygame.display.set_caption("Board")
FONT = pygame.font.SysFont("monospace",18)

iswriting=False
number_xcor,number_ycor=[],[]
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == MOUSEMOTION and iswriting:
            xcord,ycord=event.pos
            pygame.draw.circle(DISPLAY,WHITE,(xcord,ycord),4,0)
            number_xcor.append(xcord)
            number_ycor.append(ycord)
        
        if event.type == MOUSEBUTTONDOWN:
            iswriting= True
        
        if event.type == MOUSEBUTTONUP:
            iswriting=False
            number_xcor.sort()
            number_ycor.sort()
            try:
                rect_min_x,rect_max_x = max(number_xcor[0]-5,0),min(number_xcor[-1]+5,SIZEX)
                rect_min_y,rect_max_y = max(number_ycor[0]-5,0),min(number_ycor[-1]+5,SIZEX)
            except:
                continue
            number_xcor,number_ycor=[],[]
            
            img_arr=np.array(pygame.PixelArray(DISPLAY))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)
            
            if IMAGESAVE:
                cv2.imwrite("image.png",img_arr)
                imgcnt+=1
            
            if PREDICT:
                
                image=cv2.resize(img_arr,(28,28))
                image = np.pad(image,(10,10),'constant',constant_values = 0)
                image = cv2.resize(image,(28,28))//255
                
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])
                
                text = FONT.render(label,True,RED,WHITE)
                text_rect= text.get_rect()
                text_rect.left,text_rect.bottom =rect_min_x,rect_max_y
                
                DISPLAY.blit(text,text_rect)
                
        if event.type == KEYDOWN:
            DISPLAY.fill(BLACK)
    pygame.display.update()
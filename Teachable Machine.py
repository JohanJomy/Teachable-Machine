from PIL import Image as Img
from PIL import ImageTk
from tkinter import *
import numpy as np
import shutil
import cv2
import os

# initializing variables
root = Tk()      
root.title('Teachable Machine')   
root.configure(bg='black')    
root.resizable(0,0)
camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)  

root.geometry('550x550')
test = False

label = Label(root)
label.configure(bg='black')
label.grid(row = 0, column = 0)

label2 = Label(root)
label2.configure(bg='black')
label2.grid(row = 1, column = 0)

trainX = []
trainY = []

#Creating a file to store the training data or deleting and recreating the file to remove the old training data stored
path = os.getcwd()
print(path)
if not os.path.exists('Project/Training Data'):
    os.makedirs('Project/Training Data')
    os.makedirs('Project/Training Data/Object_1')
    os.makedirs('Project/Training Data/Object_2')

else:
    shutil.rmtree('Project/Training Data')
    os.makedirs('Project/Training Data')
    os.makedirs('Project/Training Data/Object 1')
    os.makedirs('Project/Training Data/Object 2')

num, num2 = 1, 1
running = True

#Function to save the images to the folder according to the label 
def save(img, label):
    global num, num2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if label == 'Object 1':
        cv2.imwrite('Project/Training Data/Object 1/' + str(num) + '.jpg', img)
        num+=1
    
    elif label == 'Object 2':
        cv2.imwrite('Project/Training Data/Object 2/'+ str(num2) + '.jpg', img)
        num2+=1

#Function to find the average color of the image
def averagecolor(image):
    return np.mean(image, axis=(0, 1))

#Function to train the model with the collected training data
def train_model():
    path = 'Project/Training Data/'
    for label in ('Object 1','Object 2'):
        print ("Loading training images for the label: "+label)
        
        for filename in os.listdir(path+label+"/"): 
            img = cv2.imread(path+label+"/"+filename)
            img_features = averagecolor(img)
            trainX.append(img_features)
            trainY.append(label)
    print('COMPLETE!')
#Function to test the model which was trained            
def test_model():
    global test
    
    #Stops the show_frame function
    test = True
    
    # making the buttons for geting the data of different objects disabled 
    button1["state"] = "disabled"
    button2["state"] = "disabled"

    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame,(int(frame.shape[1]/1.25),int(frame.shape[0]/1.25)))
    
    #Finding the object shown on the camera and give output as colored buttons(green) accoring to the distance calculated 
    features = averagecolor(frame)
    calculated_distances = []
    for i in (trainX):
        calculated_distances.append(np.linalg.norm(features-i))
        
    prediction =  trainY[np.argmin(calculated_distances)]
    
    #Making the buttons green if the object was recognized
    if prediction == 'Object 1':
        
        button1.configure(bg='green',fg='white')
        button2.configure(bg='black',fg='white')
    elif prediction == 'Object 2':
        
        button2.configure(bg='green',fg='white')
        button1.configure(bg='black',fg='white')
        
    #Shows the camera output
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)    
    img = Img.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, test_model)

#Function to get the camera reading and display it on the screen
def show_frame():
    if test == False:

        _, show_frame.frame = camera.read()
        show_frame.frame = cv2.flip(show_frame.frame, 1)
        show_frame.frame = cv2.resize(show_frame.frame,(int(show_frame.frame.shape[1]/1.25),int(show_frame.frame.shape[0]/1.25)))
        show_frame.frame = cv2.cvtColor(show_frame.frame, cv2.COLOR_BGR2RGBA)      
        
        img = Img.fromarray(show_frame.frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.after(10, show_frame)
    
show_frame()
    
class HoverButton(Button):
    def __init__(self, master, **kw):
        Button.__init__(self,master=master,**kw)
        self.configure(bg='black',fg='white', font=('Helvetica',20,'bold'),pady=15, borderwidth = 0)
        self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        
    def on_enter(self,e):
        self['background'] = 'darkslategrey'

    def on_leave(self,e):
        self['background'] = 'black'
        
#Creating buttons
button1 = HoverButton(label2,text=" OBJECT 1 ",padx=55, command = lambda: save(show_frame.frame,'Object 1'))
button1.grid(row = 0, column = 0)

button2 = HoverButton(label2,text=" OBJECT 2 ",padx=55, command = lambda: save(show_frame.frame,'Object 2'))
button2.grid(row = 0, column = 1)

train_button = HoverButton(label2,text=" Train ",padx=90, command = lambda: train_model())
train_button.grid(row = 2, column = 0)

test_button = HoverButton(label2,text=" Test ",padx=90, command = lambda: test_model())
test_button.grid(row = 2, column = 1)

root.mainloop()
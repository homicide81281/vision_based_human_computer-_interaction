# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat


from Tkinter import *
from tkColorChooser import askcolor
import subprocess as sp #create the process

status=FALSE #status varaible is used to check the program executing or not...set the default value of status variable to FALSE it means no process is running
extProc=None #to store the return value of executing program
def start() :
    global extProc #use global variable extProc
    global status #use global variable status
    if status == FALSE :#if no process is executing
        extProc = sp.Popen(['python', 'detect_blink_live.py'])  # runs detect_blink_live.py
        status = TRUE  # update the status value to TRUE means some process is executing

def stop() :
   global extProc #use global variable extProc
   global status  #use global variable status
   if status == TRUE : #if some process is running ..
        sp.Popen.terminate(extProc) # closes the process
        status = FALSE #update the value os status to FALSE to indicate no process is running


root = Tk()
root.geometry("800x500")
root.configure(background='white')
#text = Text(root)
#text.insert(INSERT, "Hello.....")
#text.insert(END, "Bye Bye.....")
#text.pack()
changeable_label = Label(root, text = 'Visual Mouse Controller' ,
    font = ('Arial' , 50), fg = 'red', width = 30, height = 2,
       borderwidth = 1, relief = 'solid',background = 'black' ).pack()
space= Label(root,  fg = 'red', width = 150, height = 2,
        background = 'grey' ).pack()
start_button = Button(master=root, text='Eye Control',command=lambda: start() ,width= 30,height=3,font = ('Arial' , 20))
start_button.pack()
quit_button = Button(master=root, text='Mouse control',command=lambda: stop(),width=30,height=3,font = ('Arial' , 20))
quit_button.pack()

root.mainloop()

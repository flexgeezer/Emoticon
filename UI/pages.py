import tkinter as tk
# first window frame startpage

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # label of frame Layout 2
        label = tk.Label(self, text="Emoticon", font=LARGEFONT)

        # putting the grid in its place by using
        # grid
        label.grid(row=0, column=2, padx=10, pady=10)

        # button to show Mode Select frame
        button1 = tk.Button(self, text="Mode Select", command=lambda: controller.show_page(Page1))

        # putting the button in its place by
        # using grid
        button1.grid(row=1, column=2, padx=100, pady=10)

        # button to show Report History frame
        button2 = tk.Button(self, text="Report History",
                            command=lambda: controller.show_page(Page2))

        # putting the button in its place by
        # using grid
        button2.grid(row=2, column=2, padx=10, pady=10)

        # button to show Report History frame
        button3 = tk.Button(self, text="Settings",
                            command=lambda: controller.show_page(Page3))

        # putting the button in its place by
        # using grid
        button3.grid(row=3, column=2, padx=10, pady=10)

        # button to show Report History frame
        button4 = tk.Button(self, text="Exit",
                            command=sys.exit)

        # putting the button in its place by
        # using grid
        button4.grid(row=0, column=0, padx=10, pady=10)


# second window frame ModeSelect
class ModeSelectPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Mode Select", font=LARGEFONT)
        label.grid(row=0, column=2, padx=10, pady=10)

        # button to show Startpage frame with text
        # layout2
        button1 = tk.Button(self, text="Main Menu",
                            command=lambda: controller.show_page(StartPage))

        # putting the button in its place
        # by using grid
        button1.grid(row=0, column=0, padx=10, pady=10)

        # button to show Live Mode frame with text
        # layout2
        button2 = tk.Button(self, text="Live Mode",command=lambda: Live_Mode())

        # putting the button in its place
        # by using grid
        button2.grid(row=1, column=2, padx=10, pady=10)

        # button to show Diagnosis frame with text
        # layout2
        button3 = tk.Button(self, text="Diagnosis Mode",
                            command=lambda: controller.show_page(Page4))

        # putting the button in its place
        # by using grid
        button3.grid(row=2, column=2, padx=10, pady=10)





class Settings(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Settings", font=LARGEFONT)
        label.grid(row=0, column=2, padx=10, pady=10)

        # button to show frame 3 with text
        # layout3
        button1 = tk.Button(self, text="Back to Main Menu",
                            command=lambda: controller.show_page(StartPage))

        # putting the button in its place by
        # using grid
        button1.grid(row=0, column=0, padx=10, pady=10)

        button2 = tk.Button(self, text="Colour Customisation")

        # putting the button in its place by
        # using grid
        button2.grid(row=1, column=2, padx=10, pady=10)

        button3 = tk.Button(self, text="Test Video Input")

        # putting the button in its place by
        # using grid
        button3.grid(row=2, column=2, padx=10, pady=10)

class InputSelect(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent,)
        label = tk.Label(self, text="Choose Input", font=LARGEFONT)
        label.grid(row=0, column=2, padx=10, pady=10)

        # button to show frame 3 with text
        # layout3
        button1 = tk.Button(self, text="Back to Mode Select",
                            command=lambda: controller.show_page(Page1))

        # putting the button in its place by
        # using grid
        button1.grid(row=0, column=0, padx=10, pady=10)

        button2 = tk.Button(self, text="Webcam",bg = "black",command = lambda:Diagnosis_Mode())

        # putting the button in its place by
        # using grid
        button2.grid(row=1, column=2, padx=10, pady=10)

        button3 = tk.Button(self, text="Video from Device")

        # putting the button in its place by
        # using grid
        button3.grid(row=2, column=2, padx=10, pady=10)

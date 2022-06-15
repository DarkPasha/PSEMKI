import tkinter as tk
#
#lass Interface(tk.Tk):
#
 #   def __init__(self, *args, **kwargs):
  #      tk.Tk.__init__(self, *args, **kwargs)
#
 #       container = tk.Frame(self, background="bisque")
  #      container.pack(fill="both", expand=True)
#
 #      
#
#interface = Interface()
#interface.minsize(600, 480)
#header_label = tk.Label(root, text = "FEHLERQUOTE:")
#header_label.grid(row = 1, column = 1)
#headerlabel = tk.Label(interface, text = "FEHLERQUOTE: ")
#headerlabel.grid(row = 1, column = 1)


#interface.mainloop()





root = tk.Tk()


header_label = tk.Label(root, text = "FEHLERQUOTE:")
header_label.grid(row = 1, column = 1)


header_label = tk.Label(root, text = "FEHLERQUOTE:")
header_label.grid(row = 1, column = 1)

root.geometry("500x200")
root['background']='#856ff8'
root.mainloop()
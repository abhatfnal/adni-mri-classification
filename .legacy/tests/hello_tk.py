# hello_tk.py

import tkinter as tk

def main():
    root = tk.Tk()
    root.title("Tkinter Hello")
    root.geometry("300x100")

    label = tk.Label(root, text="👋 Hello, Tkinter!", font=("Arial", 16))
    label.pack(expand=True)

    btn = tk.Button(root, text="Quit", command=root.destroy)
    btn.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()

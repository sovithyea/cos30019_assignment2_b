import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Traffic-Based Route Guidance System")
root.geometry("900x700")

title_lable = tk.Label(root, text="Traffic-Based Route Guidance System")
title_lable.pack()

input_frame = tk.Frame(root)
input_frame.pack(pady=20)
 
origin_lable = tk.Label(input_frame, text="Origin SCATS:")
origin_lable.grid(row=0, column=0, padx=10, pady=5)
origin_entry = tk.Entry(input_frame)
origin_entry.grid(row=0, column=1, padx=10, pady=5)

destination_lable = tk.Label(input_frame, text="Destination SCATS:")
destination_lable.grid(row=1, column=0, padx=10, pady=5)
destination_entry = tk.Entry(input_frame)
destination_entry.grid(row=1, column=1, padx=10, pady=5)

model_lable = tk.Label(input_frame, text="Select Model:")
model_lable.grid(row=2, column=0, padx=10, pady=5)  
model_dropdown = ttk.Combobox(input_frame, values=["LSTM", "GRU", "Model_3"])
model_dropdown.grid(row=2, column=1, padx=10, pady=5)

# button section
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

result_label = tk.Label(root, text="Result will appear here")
result_label.pack(pady=20)

def find_best_route():
    origin = origin_entry.get()
    destination = destination_entry.get()
    model = model_dropdown.get()
    
    result_text = f"""
        origin: {origin}
        destination: {destination}
        model: {model}
    """
    result_label.config(text=f"Origin: {origin}\nDestination: {destination}\nModel: {model}")

find_route_button = tk.Button(button_frame, text="Find Best Route", command=find_best_route)
find_route_button.pack()

root.mainloop()
    

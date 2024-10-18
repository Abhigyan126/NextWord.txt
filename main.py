import tkinter as tk
from tkinter import filedialog as fd
from collections import defaultdict
import pickle
import nltk
import threading
import tkinter.messagebox as messagebox

# Ensure NLTK resources are downloaded only once
nltk.download('punkt', quiet=True)

# File paths for loading models
BIGRAM_FILE = 'bigram_model.pkl'
TRIGRAM_FILE = 'trigram_model.pkl'

# N-gram models (bigram and trigram)
bigram_freq = defaultdict(lambda: defaultdict(int))
trigram_freq = defaultdict(lambda: defaultdict(int))

# Function to load n-gram models
def load_ngram_model():
    global bigram_freq, trigram_freq
    try:
        with open(BIGRAM_FILE, 'rb') as bf, open(TRIGRAM_FILE, 'rb') as tf:
            bigram_freq = pickle.load(bf)
            trigram_freq = pickle.load(tf)
    except FileNotFoundError:
        print("N-gram model files not found.")

# Predict next word based on n-gram models
def predict_next_word(context, top_k=1):
    words = context.lower().split()
    # Try trigram first
    if len(words) >= 2:
        last_two = (words[-2], words[-1])
        if last_two in trigram_freq:
            suggestions = sorted(trigram_freq[last_two].items(), key=lambda x: -x[1])
            return [word for word, _ in suggestions[:top_k]]
    
    # Fall back to bigram
    if len(words) >= 1:
        last_word = words[-1]
        if last_word in bigram_freq:
            suggestions = sorted(bigram_freq[last_word].items(), key=lambda x: -x[1])
            return [word for word, _ in suggestions[:top_k]]
    
    return []

# Load n-gram models at startup
load_ngram_model()

class NextWord:
    def __init__(self, root):
        self.root = root
        self.root.title("NextWord.txt")
        self.root.geometry("800x600")
        self.root.iconbitmap("icon.ico")

        # Text widget for input
        self.en1 = tk.Text(root, wrap='word', height=15, width=80, font=('Arial', 12))
        self.en1.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Variables for word prediction
        self.context = ""
        self.temporary_predicted_word = None
        self.predicted_word_position = None
        self.current_file_path = None
        self.prediction_enabled = False  # To toggle word prediction
        self.save_prompted = False  # To check if the save prompt was given

        # Bind the keypress event
        self.en1.bind("<KeyPress>", self.on_key_press)

        # Configure text styles
        self.en1.tag_configure("gray", foreground="gray")
        self.en1.tag_configure("white", foreground="white")

        # Menu setup
        self.menu = tk.Menu(root)
        root.config(menu=self.menu, bg="white")
        self.create_menu()

        # Border and color for prediction state
        self.border_color = "green"
        self.update_border()

        # Confirm quit
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # Create menu
    def create_menu(self):
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_command(label="Save As", command=self.save_as)
        file_menu.add_command(label="Exit", command=self.exit_app)

    # Create a new file
    def new_file(self):
        self.en1.delete("1.0", tk.END)
        self.root.title("Litepad - New")

    # Open an existing file
    def open_file(self):
        file_path = fd.askopenfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
                self.en1.delete("1.0", tk.END)
                self.en1.insert(tk.END, content)
            self.current_file_path = file_path
            self.root.title(f"Litepad - {file_path}")

    # Save the current file
    def save_file(self, event=None):
        if self.current_file_path:
            with open(self.current_file_path, 'w') as file:
                content = self.en1.get("1.0", tk.END)
                file.write(content)
            self.root.title(f"Litepad - {self.current_file_path}")
            self.save_prompted = False  # Reset prompt flag
        else:
            self.save_as()

    # Save the file as a new file
    def save_as(self):
        file_path = fd.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, 'w') as file:
                content = self.en1.get("1.0", tk.END)
                file.write(content)
            self.current_file_path = file_path
            self.root.title(f"Litepad - {file_path}")
            self.save_prompted = False  # Reset prompt flag

    # Exit the application
    def exit_app(self):
        if not self.save_prompted and self.en1.get("1.0", tk.END).strip():
            result = messagebox.askyesnocancel("Save Changes?", "You have unsaved changes. Do you want to save them?")
            if result:  # Save and exit
                self.save_file()
            elif result is None:  # Cancel
                return
        self.root.destroy()

    # Handle keypress for word prediction
    def on_key_press(self, event):
        char = event.char
        cursor_index = self.en1.index(tk.INSERT)
        text_before_cursor = self.en1.get("1.0", cursor_index)

        # Toggle prediction with Right Shift
        if event.keysym in ["Shift_R"]:
            self.prediction_enabled = not self.prediction_enabled
            if self.prediction_enabled:
                self.border_color = "red"  # Change color to indicate prediction is ON
            else:
                self.border_color = "green"  # Change color to indicate prediction is OFF
            self.update_border()
            return "break"

        # Accept prediction with Left Shift
        elif event.keysym in ["Shift_L"]:
            if self.temporary_predicted_word:
                self.accept_prediction(cursor_index)
            return "break"

        # Handle space key for prediction if enabled
        if self.prediction_enabled and char == " ":
            self.context = text_before_cursor.strip()
            predictions = predict_next_word(self.context)
            
            if predictions:  # If there are predictions, handle them
                self.temporary_predicted_word = predictions[0]
                threading.Thread(target=self.predict_and_display, args=(self.context,)).start()
                return "break"  # Prevent space from being added in the widget
            else:  # No prediction, insert space
                return None  # Allow space to be added to the widget

        # Remove temporary predicted word if another key is pressed
        elif self.temporary_predicted_word:
            self.en1.delete(self.predicted_word_position, f"{self.predicted_word_position} + {len(self.temporary_predicted_word) + 1} chars")
            self.temporary_predicted_word = None  # Reset prediction
        return None

    def predict_and_display(self, context):
        predictions = predict_next_word(context)
        if predictions:
            self.temporary_predicted_word = predictions[0]
            cursor_index = self.en1.index(tk.INSERT)
            self.predicted_word_position = f"{cursor_index.split('.')[0]}.{int(cursor_index.split('.')[1]) + 1}"
            self.en1.after(0, self.display_prediction)  # Use after to update UI from main thread

    def display_prediction(self):
        self.en1.insert(self.predicted_word_position, f" {self.temporary_predicted_word}", ("gray",))
        self.en1.tag_add("gray", self.predicted_word_position, f"{self.predicted_word_position} + {len(self.temporary_predicted_word)} chars")

    # Update border color based on prediction status
    def update_border(self):
        self.en1.config(highlightbackground=self.border_color, highlightcolor=self.border_color, highlightthickness=2)

    # Accept the prediction
    def accept_prediction(self, cursor_index):
        self.en1.delete(self.predicted_word_position, f"{self.predicted_word_position} + {len(self.temporary_predicted_word) + 1} chars")
        self.en1.insert(self.predicted_word_position, self.temporary_predicted_word, ("white",))
        self.temporary_predicted_word = None
        self.predicted_word_position = None

    # Confirm before closing
    def on_closing(self):
        if not self.save_prompted and self.en1.get("1.0", tk.END).strip():
            result = messagebox.askyesnocancel("Save Changes?", "You have unsaved changes. Do you want to save them?")
            if result:  # Save and exit
                self.save_file()
            elif result is None:  # Cancela
                return
        self.root.destroy()

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = NextWord(root)
    root.mainloop()

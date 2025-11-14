import tkinter as tk
from tkinter import ttk
import os

def main():
    # Create main window
    root = tk.Tk()
    root.title("Mortality Rate Comparison")
    root.geometry("1400x800")

    # Create tab control
    tab_control = ttk.Notebook(root)

    # Create two tabs
    tab_mortality_rate = ttk.Frame(tab_control)
    tab_mortality_cancer = ttk.Frame(tab_control)
    tab_edita = ttk.Frame(tab_control)

    tab_control.add(tab_mortality_rate, text="\nMortality rate Ireland/Pakistan")
    tab_control.add(tab_mortality_cancer, text="Mortality from CVD, cancer, \ndiabetes or CRD,\nages 30 and 70 (%)Rate")
    tab_control.add(tab_edita, text="\nNot implemented")
    tab_control.pack(expand=1, fill="both")

    # Get image paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mortality_rate_img_path = os.path.join(base_dir, "plots/mortality_rate_comparison.png")
    mortality_cancer_img_path = os.path.join(base_dir, "plots/mortality_cancer.png")
    edita_img_path = os.path.join(base_dir, "plots/not_implemented.png")

    # Load images
    mortality_rate_img = tk.PhotoImage(file=mortality_rate_img_path)
    mortality_cancer_img = tk.PhotoImage(file=mortality_cancer_img_path)
    edita_img = tk.PhotoImage(file=edita_img_path)

    # Display images
    mortality_rate_label = tk.Label(tab_mortality_rate, image=mortality_rate_img)
    mortality_rate_label.pack(expand=True)

    mortality_cancer_label = tk.Label(tab_mortality_cancer, image=mortality_cancer_img)
    mortality_cancer_label.pack(expand=True)

    edita_label = tk.Label(tab_edita, image=edita_img)
    edita_label.pack(expand=True)

    root.mainloop()

if __name__ == "__main__":
    main()


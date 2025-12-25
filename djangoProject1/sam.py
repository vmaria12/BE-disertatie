import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO, SAM
from tqdm import tqdm

# --- Configurare Plotting ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def select_directory(title="Selectați directorul"):
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title=title)
    return folder_selected


def select_file(title="Selectați fișierul", filetypes=[("All Files", "*.*")]):
    root = tk.Tk()
    root.withdraw()
    file_selected = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_selected


def compute_tumor_size(mask):
    """Calculează dimensiunea tumorii (număr pixeli)."""
    return np.sum(mask > 0)


def compute_curvature(mask):
    """
    Estimează curbura sau complexitatea formei.
    Metrică: (Perimetru^2) / (4 * pi * Aria).
    """
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area == 0:
        return 0

    complexity = (perimeter ** 2) / (4 * np.pi * area)
    return complexity


def run_evaluation():
    print("=== Script Segmentare și Analiză Formă (Fără Ground Truth) ===")

    # 1. Selectare Resurse
    print("\n[Pasul 1] Selectați modelul YOLO pentru detecție (Box Prompt)...")
    yolo_path = select_file("Selectați modelul YOLO (.pt)", [("YOLO Model", "*.pt")])
    if not yolo_path: return

    print("\n[Pasul 2] Selectați modelul SAM pentru segmentare...")
    sam_path = select_file("Selectați modelul SAM (.pt)", [("SAM Model", "*.pt")])
    if not sam_path: return

    print("\n[Pasul 3] Selectați directorul cu IMAGINI de analizat...")
    images_dir = select_directory("Director Imagini")
    if not images_dir: return

    # Setup Output
    output_vis_dir = os.path.join(os.path.dirname(__file__), "vizualizare_segmentare")
    os.makedirs(output_vis_dir, exist_ok=True)
    print(f"\nImaginile segmentate vor fi salvate în: {output_vis_dir}")

    # Inițializare Modele
    print(f"\nÎncărcare modele...\n YOLO: {os.path.basename(yolo_path)}\n SAM: {os.path.basename(sam_path)}")
    try:
        yolo_model = YOLO(yolo_path)
        sam_model = SAM(sam_path)
    except Exception as e:
        print(f"Eroare la încărcarea modelelor: {e}")
        return

    results = []
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    print(f"\nÎncepere analiză pe {len(image_files)} imagini...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(images_dir, img_name)

        image = cv2.imread(img_path)
        if image is None: continue

        # A. Detecție cu YOLO
        results_yolo = yolo_model(image, verbose=False)

        predicted_size = 0
        predicted_complexity = 0
        has_detection = False

        if results_yolo[0].boxes:
            best_box = results_yolo[0].boxes[0]
            bbox = best_box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]

            # B. Segmentare cu SAM
            results_sam = sam_model(image, bboxes=[bbox], verbose=False)

            if results_sam[0].masks:
                mask_pred = results_sam[0].masks.data[0].cpu().numpy()
                mask_pred_uint8 = (mask_pred * 255).astype(np.uint8)

                # Resize la original pt calcul metrici corecte
                if mask_pred_uint8.shape[:2] != image.shape[:2]:
                    mask_pred_uint8 = cv2.resize(mask_pred_uint8, (image.shape[1], image.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)

                # Calcul metrici pe predicție
                predicted_size = compute_tumor_size(mask_pred_uint8)
                predicted_complexity = compute_curvature(mask_pred_uint8)
                has_detection = True

                # C. Salvare Vizualizare (Overlay)
                # Creăm un overlay roșu transparent
                overlay = image.copy()
                overlay[mask_pred_uint8 > 0] = [0, 0, 255]  # Red mask
                cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

                # Desenăm și box-ul YOLO cu verde
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

                out_path = os.path.join(output_vis_dir, f"seg_{img_name}")
                cv2.imwrite(out_path, image)

        results.append({
            'Image': img_name,
            'Detected': has_detection,
            'Predicted_Tumor_Size': predicted_size,
            'Predicted_Complexity': predicted_complexity
        })

    if not results:
        print("Nu s-au generat rezultate.")
        return

    df = pd.DataFrame(results)

    # 2. Salvare CSV
    output_csv = os.path.join(os.path.dirname(__file__), "rezultate_analiza_segmentare.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nDatele detaliate au fost salvate în {output_csv}")

    # Filtrare doar detectate pentru grafice
    df_det = df[df['Detected'] == True]

    if df_det.empty:
        print("Nu s-au detectat tumori în nicio imagine.")
        return

    print(f"\nTumori detectate în {len(df_det)} din {len(df)} imagini.")
    print(f"Dimensiune medie tumoare: {df_det['Predicted_Tumor_Size'].mean():.2f} pixeli")
    print(f"Complexitate medie formă: {df_det['Predicted_Complexity'].mean():.2f}")

    plots_dir = os.path.join(os.path.dirname(__file__), "grafice_segmentare")
    os.makedirs(plots_dir, exist_ok=True)

    # D. Histograms & Scatter
    plt.figure()
    sns.histplot(df_det['Predicted_Tumor_Size'], kde=True, color='purple')
    plt.title("Distribuția Dimensiunilor Tumorilor Detectate")
    plt.xlabel("Dimensiune (pixeli)")
    plt.savefig(os.path.join(plots_dir, "distributie_marime.png"))
    plt.close()

    plt.figure()
    sns.scatterplot(data=df_det, x='Predicted_Tumor_Size', y='Predicted_Complexity', hue='Predicted_Complexity',
                    palette='coolwarm')
    plt.title("Dimensiune Tumoare vs. Complexitate Formă")
    plt.xlabel("Dimensiune (pixeli)")
    plt.ylabel("Complexitate (1 = Cerc, >1 = Iregulat)")
    plt.savefig(os.path.join(plots_dir, "size_vs_complexity.png"))
    plt.close()

    print(f"\nGraficele au fost salvate în: {plots_dir}")
    print("Script finalizat.")


if __name__ == "__main__":
    run_evaluation()

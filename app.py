import gradio as gr
import numpy as np
from PIL import Image
import time

# ==========================================
# 1. VECTORIZED ALGORITHM (Fast)
# ==========================================
def vectorized_mosaic(img_array, grid_size, tile_array, base_colors):
    height, width, channels = img_array.shape
    tile_h, tile_w = height // grid_size, width // grid_size
    
    cropped = img_array[:tile_h * grid_size, :tile_w * grid_size, :]
    reshaped = cropped.reshape(grid_size, tile_h, grid_size, tile_w, channels)
    
    avg_colors = reshaped.mean(axis=(1, 3)).astype(np.uint8) 
    
    distances = np.linalg.norm(avg_colors[:, :, None] - base_colors, axis=3)
    closest_tile_indices = np.argmin(distances, axis=2)
    
    mosaic_5d = tile_array[closest_tile_indices]
    final_mosaic = mosaic_5d.swapaxes(1, 2).reshape(grid_size * tile_array.shape[1], grid_size * tile_array.shape[2], 3)
    
    return final_mosaic, cropped, avg_colors

# ==========================================
# 2. NAIVE LOOP ALGORITHM (Slow - For Step 6)
# ==========================================
def loop_based_mosaic(img_array, grid_size, tile_array, base_colors):
    height, width, channels = img_array.shape
    tile_h, tile_w = height // grid_size, width // grid_size
    display_tile_size = tile_array.shape[1]
    
    final_img = np.zeros((grid_size * display_tile_size, grid_size * display_tile_size, 3), dtype=np.uint8)
    
    for r in range(grid_size):
        for c in range(grid_size):
            y_start, y_end = r * tile_h, (r + 1) * tile_h
            x_start, x_end = c * tile_w, (c + 1) * tile_w
            region = img_array[y_start:y_end, x_start:x_end]
            
            avg_color = np.mean(region, axis=(0, 1))
            
            best_idx = 0
            min_dist = float('inf')
            for i, color in enumerate(base_colors):
                dist = np.linalg.norm(avg_color - color)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
                    
            paste_y = r * display_tile_size
            paste_x = c * display_tile_size
            final_img[paste_y:paste_y+display_tile_size, paste_x:paste_x+display_tile_size] = tile_array[best_idx]
            
    return final_img

# ==========================================
# 3. PIPELINE & GRADIO INTERFACE
# ==========================================
def process_image(input_img, grid_size, num_tiles):
    img_array = np.array(input_img, dtype=np.uint8)
    
    # Prepare tile set
    tile_size = 20
    np.random.seed(42)
    base_colors = np.random.randint(0, 255, size=(int(num_tiles), 3), dtype=np.uint8)
    tile_array = np.zeros((int(num_tiles), tile_size, tile_size, 3), dtype=np.uint8)
    for i in range(int(num_tiles)):
        tile_array[i, :, :] = base_colors[i]
        tile_array[i, 0, :] = 20
        tile_array[i, :, 0] = 20
        
    # Measure Vectorized Time
    start_vec = time.time()
    vectorized_result, cropped_orig, avg_colors = vectorized_mosaic(img_array, grid_size, tile_array, base_colors)
    time_vec = time.time() - start_vec
    
    # Measure Loop Time (Your missing requirement)
    start_loop = time.time()
    _ = loop_based_mosaic(img_array, grid_size, tile_array, base_colors)
    time_loop = time.time() - start_loop
    
    # Format Outputs
    canvas_size = grid_size * tile_size
    segmented_preview = Image.fromarray(avg_colors, 'RGB').resize((canvas_size, canvas_size), Image.NEAREST)
    
    mosaic_img = Image.fromarray(vectorized_result)
    original_resized = Image.fromarray(cropped_orig).resize((canvas_size, canvas_size))
    final_artistic_mosaic = Image.blend(mosaic_img, original_resized, alpha=0.25)
    
    # Calculate MSE Metric
    orig_resized_arr = np.array(original_resized, dtype=np.float32)
    result_arr = np.array(final_artistic_mosaic, dtype=np.float32)
    mse_value = np.mean((orig_resized_arr - result_arr) ** 2)
    
    # Generate the Lab Report with both times
    report = (
        f"--- Performance Metric ---\n"
        f"Mean Squared Error (MSE): {mse_value:.2f}\n\n"
        f"--- Computational Performance (Step 6) ---\n"
        f"Grid Size: {grid_size}x{grid_size}\n"
        f"Vectorized Processing Time: {time_vec:.5f} seconds\n"
        f"Loop-Based Processing Time: {time_loop:.5f} seconds\n"
        f"Speedup Factor: {time_loop / time_vec:.2f}x faster using NumPy"
    )
    
    return segmented_preview, final_artistic_mosaic, report

demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Test Image"),
        gr.Slider(minimum=16, maximum=64, step=16, value=32, label="Grid Size (NxN)"),
        gr.Slider(minimum=2, maximum=32, step=2, value=8, label="Tile Set Palette Size")
    ],
    outputs=[
        gr.Image(label="1. Segmented Grid Preview"),
        gr.Image(label="2. Final Blended Mosaic"),
        gr.Textbox(label="Assignment Report & Metrics", lines=9)
    ],
    title="Interactive Image Mosaic Generator",
    description="Fulfills all requirements including vector vs loop timing, MSE metrics, and UI."
)

if __name__ == "__main__":
    demo.launch()

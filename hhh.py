from tkinter import Tk, Button, filedialog, Canvas
from PIL import ImageTk, Image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from tkinter import Label
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
def predict_image(image_path):
    # Tải mô hình đã lưu
    loaded_model = load_model('trained_model.h5')
    # Chuẩn bị ảnh để dự đoán
    test_image = load_img(image_path, target_size=(32, 32))
    test_image = img_to_array(test_image)
    test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
    test_image = test_image.astype('float32')
    test_image /= 255.0
    # Dự đoán lớp của ảnh
    classes = ['cano', 'may_bay', 'may_keo', 'tau', 'tau_dien_ngam', 'tau_hoa', 'thuyen', 'truc_thang', 'xe_ba_gac', 'xe_buyt', 'xe_cau', 'xe_cuu_hoa', 'xe_cuu_thuong', 'xe_dap', 'xe_dien', 'xe_may', 'xe_oto', 'xe_tai', 'xe_tai_nho']
    result = loaded_model.predict(test_image)
    predicted_class = classes[np.argmax(result)]
    return predicted_class
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((300, 400))
        # Hiển thị ảnh trên canvas
        canvas.image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor='nw', image=canvas.image)
        # Dự đoán lớp của ảnh
        predicted_class = predict_image(file_path)
        # Hiển thị kết quả dự đoán
        result_label.configure(text="Đây là: " + predicted_class)
        # Hiển thị canvas
        canvas.pack()
        result_label.pack()
# Tạo cửa sổ gốc
root = Tk()
root.title("Nhận biết và phân loại các loại phương")
new_logo_path = "logo.png"
root.iconphoto(False, ImageTk.PhotoImage(Image.open(new_logo_path)))
# Tạo nút "Tải ảnh lên"
button = Button(root, text="Tải ảnh lên", command=load_image)
button.pack()
# Tạo canvas để hiển thị ảnh
canvas = Canvas(root, width=300, height=400)
# Tạo nhãn để hiển thị kết quả dự đoán
result_label = Label(root, text="")
root.mainloop()
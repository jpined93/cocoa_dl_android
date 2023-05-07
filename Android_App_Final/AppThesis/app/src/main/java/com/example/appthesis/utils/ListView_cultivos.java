package com.example.appthesis.utils;

import android.graphics.Bitmap;

public class ListView_cultivos {

    private String image_name;
    private String diagnostico;
    private String fechaDate;
    private byte[] image;
    private String id_Foto;

    public ListView_cultivos(String image_name, String diagnostico, String fechaDate, byte[] image, String id_Foto) {
        this.image_name = image_name;
        this.diagnostico = diagnostico;
        this.fechaDate = fechaDate;
        this.image=image;
        this.id_Foto=id_Foto;
    }

    public String getImage_name() {
        return image_name;
    }

    public void setImage_name(String image_name) {
        this.image_name = image_name;
    }

    public String getDiagnostico() {
        return diagnostico;
    }

    public void setDiagnostico(String diagnostico) {
        this.diagnostico = diagnostico;
    }

    public String getFechaDate() {
        return fechaDate;
    }

    public void setFechaDate(String fechaDate) {
        this.fechaDate = fechaDate;
    }

    public byte[] getImage() {
        return image;
    }

    public void setImage(byte[] image) {
        this.image = image;
    }

    public String getId_Foto() {
        return id_Foto;
    }

    public void setId_Foto(String id_Foto) {
        this.id_Foto = id_Foto;
    }
}

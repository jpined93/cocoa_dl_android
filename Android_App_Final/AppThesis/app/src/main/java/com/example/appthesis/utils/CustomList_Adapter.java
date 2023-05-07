package com.example.appthesis.utils;

import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;

import com.example.appthesis.R;
import com.example.appthesis.services.PermissionService;
import com.squareup.picasso.Picasso;

import java.io.File;
import java.util.ArrayList;

import javax.inject.Inject;

public class CustomList_Adapter  extends ArrayAdapter<ListView_cultivos> {

    private static final String LOG_TAG = CustomList_Adapter.class.getSimpleName();
    public CustomList_Adapter(Activity context, ArrayList<ListView_cultivos> cultivo) {
        super(context, 0, cultivo);
    }

    @Inject
    PermissionService permissionService;


    @Override
    public View getView(int position, View convertView, ViewGroup parent) {


        View listItemView = convertView;
        if (convertView==null){
            LayoutInflater layoutInflater = (LayoutInflater) getContext().getSystemService(Activity.LAYOUT_INFLATER_SERVICE);
            convertView = layoutInflater.inflate(R.layout.custom_view,parent,false);
        }

        ListView_cultivos cultivo_s =getItem(position);


        ImageView imageView= (ImageView) convertView.findViewById(R.id.Image_Flag);
        byte[] imageCacao = cultivo_s.getImage();
        Bitmap bitmap = BitmapFactory.decodeByteArray(imageCacao, 0, imageCacao.length);
        imageView.setImageBitmap(bitmap);


        TextView txtNameImage=(TextView) convertView.findViewById(R.id.txtNameImage);
        txtNameImage.setText(cultivo_s.getImage_name());

        TextView txtFecha=(TextView) convertView.findViewById(R.id.txtFecha);
        txtFecha.setText(cultivo_s.getFechaDate());

        TextView txtDiagnostico=(TextView) convertView.findViewById(R.id.txtDiagnostico);
        txtDiagnostico.setText(cultivo_s.getDiagnostico());

        TextView txtId=(TextView) convertView.findViewById(R.id.txtid_num);
        txtId.setText(cultivo_s.getId_Foto());

        return convertView;
    }

}

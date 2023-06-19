package com.example.appthesis.activities;
import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.example.appthesis.services.AdminSQLiteOpenHelper;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.MarkerOptions;

import androidx.annotation.NonNull;

import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import androidx.fragment.app.FragmentActivity;

import com.example.appthesis.databinding.ActivityImagenCacaoBinding;
import com.example.appthesis.R;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;

import lombok.val;

public class ImagenCacao extends FragmentActivity implements OnMapReadyCallback {
    private ActivityImagenCacaoBinding binding;
    Integer i;
    Double lat, lon;
    private GoogleMap mMap;
    String NameI, FechaI;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityImagenCacaoBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        String dato=getIntent().getStringExtra("dato");
        i= Integer.parseInt(dato);

        SupportMapFragment mapFragment = (SupportMapFragment) getSupportFragmentManager()
                .findFragmentById(R.id.map1);
        mapFragment.getMapAsync(this);

        binding.btnInvestigacion.setOnClickListener(view -> {
            Investiga();
        });
        binding.btnServicio.setOnClickListener(v ->{
            Evaluar();
        });


    }

    public void Fill_view(){
        AdminSQLiteOpenHelper admin = new AdminSQLiteOpenHelper(this, "administracion", null, 1);
        SQLiteDatabase BaseDeDatabase = admin.getWritableDatabase();
        String query1=  String.format("select * from CACAO Where id= %s", i);
        Cursor fila = BaseDeDatabase.rawQuery(query1, null);


        if(fila.moveToLast()){
            do {/*
                Id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "name VARCHAR, diagnostico VARCHAR, fecha VARCHAR, latitud VARCHAR, longitud VARCHAR, notas VARCHAR, image BLOB"*/
            binding.txtNameImageI.setText(fila.getString(1));
            NameI=fila.getString(1);
            FechaI=fila.getString(3);
            binding.txtFechaI.setText(fila.getString(3));
            binding.txtDiagnosticoI.setText(fila.getString(2));
            binding.txtNotasI.setText(fila.getString(6));
            byte[] imageCacao = fila.getBlob(7);
            Bitmap bitmap = BitmapFactory.decodeByteArray(imageCacao, 0, imageCacao.length);
            binding.ImageFlag.setImageBitmap(bitmap);
            lat=Double.valueOf(fila.getString(4));
            lon=Double.valueOf(fila.getString(5));
            }while (fila.moveToPrevious());
            fila.close();
            fila=null;
            BaseDeDatabase.close();
        } else {
            Toast.makeText(this,"No existen registros actualmente", Toast.LENGTH_SHORT).show();
            BaseDeDatabase.close();
        }

    }
    public void Evaluar(){
        Bitmap imageBitmap;
        imageBitmap=((BitmapDrawable)binding.ImageFlag.getDrawable()).getBitmap();
        int dimension = Math.min(imageBitmap.getWidth(), imageBitmap.getHeight());
        imageBitmap = Bitmap.createScaledBitmap(imageBitmap, 300, 300, false);
        imageBitmap = ThumbnailUtils.extractThumbnail(imageBitmap, dimension, dimension);
        String json_string=getStringFromBitmap(imageBitmap);
        postDataUsingVolley(json_string,"job");
    }


    private void postDataUsingVolley(String imagen, String job) {
        String url = "http://34.125.196.167:5000/uImg";//+"?img="+"\"" + name + "\"";
        RequestQueue queue = Volley.newRequestQueue(this);
        StringRequest request = new StringRequest(Request.Method.POST, url, new com.android.volley.Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                Toast.makeText(ImagenCacao.this, "Data added to API", Toast.LENGTH_SHORT).show();
                try {
                    JSONObject respObj = new JSONObject(response);
                    String name = respObj.getString("img");
                    System.out.println(name);
                    binding.txtDiagnosticoI.setText(name);
                } catch (JSONException e) {
                    e.printStackTrace();}}
        }, new com.android.volley.Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(ImagenCacao.this, "Fail to get response = " + error, Toast.LENGTH_SHORT).show();
            }
        }) {
            @Override
            public byte[] getBody() throws AuthFailureError {
                System.out.println(imagen.getBytes());
                return imagen.getBytes();
            }
        };
        queue.add(request);
    }



    private String getStringFromBitmap(Bitmap bitmapPicture) {
        final int COMPRESSION_QUALITY = 100;
        String encodedImage;
        ByteArrayOutputStream byteArrayBitmapStream = new ByteArrayOutputStream();
        bitmapPicture.compress(Bitmap.CompressFormat.PNG, COMPRESSION_QUALITY,
                byteArrayBitmapStream);
        byte[] b = byteArrayBitmapStream.toByteArray();
        encodedImage = Base64.encodeToString(b, Base64.DEFAULT);
        Log.d("image_string",encodedImage);
        return encodedImage;
    }



    @Override
    public void onMapReady(@NonNull GoogleMap googleMap) {
        Fill_view();
        mMap = googleMap;
        LatLng NewPoint = new LatLng(lat, lon);
        mMap.addMarker(new MarkerOptions().position(NewPoint).title(NameI).snippet(FechaI));
        mMap.moveCamera(CameraUpdateFactory.newLatLng(NewPoint));
        mMap.animateCamera(CameraUpdateFactory.newLatLngZoom(NewPoint,20.0f));
    }
    public void actualizar(View view) {
        AdminSQLiteOpenHelper admin = new AdminSQLiteOpenHelper(this, "administracion", null, 1);
        String result;
        result= binding.txtDiagnosticoI.getText().toString();
        admin.updateData(result, String.valueOf(i));
        Toast.makeText(this, "Actualización Exitosa", Toast.LENGTH_SHORT).show();
    }

    public void Investiga() {
        String Diagnostico = binding.txtDiagnosticoI.getText().toString();
        String webpage;
        if (Diagnostico.equals("Sano")){webpage="";}
        else if (Diagnostico.equals("Monoliosis")){webpage="https://www.agrosavia.co/media/11540/69317.pdf";}
        else if (Diagnostico.equals("Mazorca Negra")){webpage="http://www.fhia.org.hn/descargas/proyecto_procacao/infocacao/InfoCacao_No13_Jul_2017.pdf";}
        else if (Diagnostico.equals("Lasiodiplodia")){webpage="http://repositorio.uaaan.mx:8080/xmlui/bitstream/handle/123456789/46232/Gonz%C3%A1lez%20Ru%C3%ADz%20Aide%C3%A9.pdf?sequence=1&isAllowed=y";}
        else {
            webpage="No encontrado";
        }
        if (!webpage.equals("No encontrado")) {
            Intent web = new Intent(ImagenCacao.this, wbview.class);
            web.putExtra("dato", webpage);
            startActivity(web);
        }
    }
}
package com.example.appthesis.activities;

import static java.text.DateFormat.getDateInstance;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.FileProvider;

import android.annotation.SuppressLint;
import android.content.ActivityNotFoundException;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.location.Location;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.example.appthesis.BuildConfig;
import com.example.appthesis.databinding.ActivityCameraBinding;
import com.example.appthesis.services.AdminSQLiteOpenHelper;
import com.example.appthesis.services.PermissionService;
import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.location.Priority;
import com.google.android.gms.tasks.OnSuccessListener;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.util.Date;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

import javax.inject.Inject;

import dagger.hilt.android.AndroidEntryPoint;

@AndroidEntryPoint
public class CameraActivity extends BaseActivity {
    public static final String TAG = CameraActivity.class.getName();
    ActivityCameraBinding binding;
    @Inject
    PermissionService permissionService;
    ImageView Image1;
    static final int REQUEST_IMAGE_CAPTURE = 1;
    static final int REQUEST_EXTERNAL_STORAGE = 3;
    private Uri pictureImagePath = null;
    public static final int DEFAULT_UPDATE_INTERNAL = 10;
    public static final int FAST_UPDATE_INTERVAL = 5;
    LocationRequest locationRequest;
    LocationCallback locationCallback;
    // Google's API for location services. The majority of the app functions using this class,
    FusedLocationProviderClient fusedLocationProviderClient;
    Location currentLocation;
    String mCurrentPhotoPath;
    Bitmap imageBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityCameraBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        locationRequest = createLocationRequest();
        fusedLocationProviderClient = LocationServices.getFusedLocationProviderClient(CameraActivity.this);
        locationCallback = new LocationCallback() {
            @Override
            public void onLocationResult(@NonNull LocationResult locationResult) {
                super.onLocationResult(locationResult);
                updateIValues(locationResult.getLastLocation());
            }
        };

        binding.buttonCamera.setOnClickListener(v -> {

                    if (!permissionService.ismCameraPermissionGranted()) {
                        permissionService.getCameraPermission(this);
                    } else {
                        takePictureOrVideo();
                    }
                }


        );
           updateGPS();

           binding.buttonGallery.setOnClickListener(v ->{
               Evaluar();
           });
    }

    String currentPhotoPath;

    private void stopLocationUpdates() {
        fusedLocationProviderClient.removeLocationUpdates(locationCallback);
    }

    @SuppressLint("MissingPermission")
    private void startLocationUpdates() {
        if (permissionService.ismLocationPermissionGranted()) {
            fusedLocationProviderClient.requestLocationUpdates(locationRequest, locationCallback, null);
            updateGPS();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == PermissionService.PERMISSIONS_REQUEST_CAMERA) {
            permissionService.getCameraPermission(this);
            if (permissionService.ismCameraPermissionGranted() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                takePictureOrVideo();
            } else {
                alertsHelper.longSimpleSnackbar(binding.getRoot(), "No se pudo obtener el pemiso para acceder a la camara");
            }
        } else if (requestCode == PermissionService.PERMISSIONS_REQUEST_LOCATION) {
            permissionService.getLocationPermission(this);
            if (permissionService.ismLocationPermissionGranted()) {
                updateGPS();
            } else {
                Toast.makeText(this, "Necesita permiso", Toast.LENGTH_SHORT).show();
            }
        }
    }

    protected LocationRequest createLocationRequest() {
        return LocationRequest.create()
                .setInterval(100 * DEFAULT_UPDATE_INTERNAL)
                .setFastestInterval(100 * FAST_UPDATE_INTERVAL)
                .setMaxWaitTime(TimeUnit.SECONDS.toMillis(10))
                .setPriority(Priority.PRIORITY_BALANCED_POWER_ACCURACY);
    }

    @Override
    protected void onResume() {
        super.onResume();
        permissionService.getLocationPermission(this);
        if (permissionService.ismLocationPermissionGranted()) {
            updateGPS();
        }
    }


    @SuppressLint("MissingPermission")
    private void updateGPS() {
        if (permissionService.ismLocationPermissionGranted()) {
            fusedLocationProviderClient.getLastLocation().addOnSuccessListener(this, new OnSuccessListener<Location>() {
                @Override
                public void onSuccess(Location location) {
                    updateIValues(location);
                    currentLocation = location;
                }
            });
        } else {
            Toast.makeText(this, "Esta app necesita permiso", Toast.LENGTH_SHORT).show();
            permissionService.getLocationPermission(this);
        }
    }

    private void updateIValues(Location location) {
        binding.txtLatitud.setText(String.valueOf(location.getLatitude()));
        binding.txtLongitud.setText(String.valueOf(location.getLongitude()));
    }


    private void takePictureOrVideo() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        //Create temp file for image result
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = String.format("%s.jpg", timeStamp);

        binding.txtUbicacion.setText(String.format("Image %s", timeStamp));
        binding.txtDiagnostico.setText("Diagnóstico Pendiente");
        binding.txtnameImage.setText(String.format("Image %s", timeStamp));
        startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            binding.txtDiagnostico.setText("Diagnóstico Pendiente");
            binding.preview.removeAllViews();
            ImageView imageView = new ImageView(CameraActivity.this);
            imageView.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT));
            Bundle extras = data.getExtras();
            imageBitmap = (Bitmap) extras.get("data");
            imageView.setImageBitmap(imageBitmap);
            Image1 = imageView;
            imageView.setRotation(90f);
            /*imageView.setImageURI(pictureImagePath);*/
            imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
            imageView.setAdjustViewBounds(true);
            binding.preview.addView(imageView);
            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            /* String timeStamp = getDateInstance().format(new Date());*/
            binding.txtFecha.setText(timeStamp);
        }
    }
    public void Registrar(View view) {
        AdminSQLiteOpenHelper admin = new AdminSQLiteOpenHelper(this, "administracion", null, 1);
        String nameImage = binding.txtnameImage.getText().toString();
        String Diagnostico = binding.txtDiagnostico.getText().toString();
        String Fecha = binding.txtFecha.getText().toString();
        String Latitud = binding.txtLatitud.getText().toString();
        String Longitud = binding.txtLongitud.getText().toString();
        String Notas = binding.txtNotas.getText().toString();
        if (!nameImage.isEmpty() && !Latitud.isEmpty() && !Longitud.isEmpty()) {
            if (Notas.isEmpty()) {
                Notas = "NA";
            }
            admin.insertData(nameImage, Diagnostico, Fecha, Latitud, Longitud, Notas, imageViewToByte(Image1));
            binding.preview.removeAllViews();
            binding.txtnameImage.setText("");
            Toast.makeText(this, "Registro exitoso", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Debes llenar todos los campos", Toast.LENGTH_SHORT).show();
        }
    }
    public void Evaluar(){
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
                Toast.makeText(CameraActivity.this, "Data added to API", Toast.LENGTH_SHORT).show();
                try {
                    JSONObject respObj = new JSONObject(response);
                    String name = respObj.getString("img");
                    System.out.println(name);
                    binding.txtDiagnostico.setText(name);
                } catch (JSONException e) {
                    e.printStackTrace();}}
        }, new com.android.volley.Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(CameraActivity.this, "Fail to get response = " + error, Toast.LENGTH_SHORT).show();
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

    public static byte[] imageViewToByte(ImageView image) {
        Bitmap bitmap = ((BitmapDrawable)image.getDrawable()).getBitmap();
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
        byte[] byteArray = stream.toByteArray();
        return byteArray;
    }
}


package com.example.nnapp;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.Image;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;


import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

public class MainActivity extends AppCompatActivity {
    private Button btnTakePicture,btnLaunchGallery;
    private ImageView imgLoaded;
    private TextView txtClassif;
    int imageSize=300;

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){

            if(requestCode == 3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imgLoaded.setImageBitmap(image);
                System.out.println("Image loaded in frame");
                //image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                String json_string=getStringFromBitmap(image);
                System.out.println("Image transform to text");
                //System.out.println(json_string);
                //postDataUsingVolley(json_string,"job");
                serviceHealthCheck("job");

            }else{
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imgLoaded.setImageBitmap(image);

                //image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                String json_string=getStringFromBitmap(image);
                System.out.println("Image transform to text");
                //System.out.println(json_string);
                //postDataUsingVolley(json_string,"job");
                serviceHealthCheck("job");
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
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

    private  void serviceHealthCheck(String job) {
        // url to post our data
        RequestQueue queue = Volley.newRequestQueue(this);
        String url = "http://34.125.196.167/uImg";//+"?img="+"\"" + name + "\"";
        // Request a string response from the provided URL.
        StringRequest stringRequest = new StringRequest(Request.Method.GET, url,
                new Response.Listener<String>() {
                    @Override
                    public void onResponse(String response) {
                        // Display the first 500 characters of the response string.
                        Log.d("image_string","Response is: " + response.substring(0,500));
                        Toast.makeText(MainActivity.this, "Data added to API", Toast.LENGTH_SHORT).show();

                    }
                }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Log.d("image_string","Failed with error"+error);
                Toast.makeText(MainActivity.this, "Failed with error"+error.networkResponse.statusCode, Toast.LENGTH_SHORT).show();
            }
        });

// Add the request to the RequestQueue.
        queue.add(stringRequest);

    }

    private void postDataUsingVolley(String name, String job) {
        // url to post our data
        String url = "http://34.125.196.167/uImg";//+"?img="+"\"" + name + "\"";

        // creating a new variable for our request queue
        RequestQueue queue = Volley.newRequestQueue(MainActivity.this);

        // on below line we are calling a string
        // request method to post the data to our API
        // in this we are calling a post method.
        StringRequest request = new StringRequest(Request.Method.POST, url, new com.android.volley.Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                // inside on response method we are
                // hiding our progress bar
                // and setting data to edit text as empty

                // on below line we are displaying a success toast message.
                Toast.makeText(MainActivity.this, "Data added to API", Toast.LENGTH_SHORT).show();

                try {
                    // on below line we are parsing the response
                    // to json object to extract data from it.
                    JSONObject respObj = new JSONObject(response);

                    // below are the strings which we
                    // extract from our json object.
                    String name = respObj.getString("img");

                    // on below line we are setting this string s to our text view.
                    System.out.println(name);
                    txtClassif.setText("Classified as "+name);
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }
        }, new com.android.volley.Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(MainActivity.this, "Fail to get response = " + error, Toast.LENGTH_SHORT).show();
            }

        }) {
            /*@Override
            protected Map<String, String> getParams() {
                // below line we are creating a map for
                // storing our values in key and value pair.
                Map<String, String> params = new HashMap<String, String>();

                // on below line we are passing our key
                // and value pair to our parameters.
                params.put("name", name);
                params.put("job", job);

                // at last we are
                // returning our params.
                System.out.println(params);
                return params;
            }*/

            @Override
            public byte[] getBody() throws AuthFailureError {
                System.out.println(name.getBytes());
                return name.getBytes();
            }
        };
        // below line is to make
        // a json object request.

        queue.add(request);
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnTakePicture=findViewById(R.id.btnTakePicture);
        btnLaunchGallery=findViewById(R.id.btnLaunchGallery);

        imgLoaded=findViewById(R.id.imgLoaded);
        txtClassif=findViewById(R.id.txtClassif);

        btnTakePicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(checkSelfPermission(Manifest.permission.CAMERA)== PackageManager.PERMISSION_GRANTED){
                    Intent cameraIntent  = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent,3);

                }else{
                    requestPermissions(new String[]{Manifest.permission.CAMERA},100);
                }
            }
        });

        btnLaunchGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent  = new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent,1);
            }
        });


    }
}
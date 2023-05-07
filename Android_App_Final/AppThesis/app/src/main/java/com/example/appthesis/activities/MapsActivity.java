package com.example.appthesis.activities;

import static java.security.AccessController.getContext;

import androidx.fragment.app.FragmentActivity;

import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.widget.Toast;
import com.example.appthesis.R;
import com.example.appthesis.services.AdminSQLiteOpenHelper;
import com.example.appthesis.utils.ListView_cultivos;
import com.example.appthesis.utils.BitmapUtils;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.BitmapDescriptor;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.MarkerOptions;
import com.example.appthesis.databinding.ActivityMapsBinding;

public class MapsActivity extends FragmentActivity implements OnMapReadyCallback {

    private GoogleMap mMap;
    private ActivityMapsBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMapsBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Obtain the SupportMapFragment and get notified when the map is ready to be used.
        SupportMapFragment mapFragment = (SupportMapFragment) getSupportFragmentManager()
                .findFragmentById(R.id.map);
        mapFragment.getMapAsync(this);
    }

    /**
     * Manipulates the map once available.
     * This callback is triggered when the map is ready to be used.
     * This is where we can add markers or lines, add listeners or move the camera. In this case,
     * we just add a marker near Sydney, Australia.
     * If Google Play services is not installed on the device, the user will be prompted to install
     * it inside the SupportMapFragment. This method will only be triggered once the user has
     * installed Google Play services and returned to the app.
     */
    @Override
    public void onMapReady(GoogleMap googleMap) {
        mMap = googleMap;

        AdminSQLiteOpenHelper admin = new AdminSQLiteOpenHelper(this, "administracion", null, 1);
        SQLiteDatabase BaseDeDatabase = admin.getWritableDatabase();
        Cursor fila = BaseDeDatabase.rawQuery
                ("select name,latitud, longitud , diagnostico from CACAO ", null);
        // Add a marker in Sydney and move the camera
        if(fila.moveToLast()){
            LatLng NewPoint;
            do {
                 NewPoint = new LatLng(Double.valueOf(fila.getString(1)), Double.valueOf(fila.getString(2)));
                 int colormakr;
                 if (fila.getString(3).equals("Diagnóstico Pendiente")){colormakr=R.drawable.grey_baseline_health_and_safety_24;}
                 else if (fila.getString(3).equals("Sano")) {colormakr=R.drawable.green_baseline_health_and_safety_24;}
                 else {colormakr=R.drawable.red_baseline_health_and_safety_24;}
                mMap.addMarker(new MarkerOptions().position(NewPoint)
                        .title(fila.getString(0))
                        .icon(BitmapUtils.getBitmapDescriptor(getApplicationContext(), colormakr))
                        .snippet(fila.getString(3)));
            }while (fila.moveToPrevious());
            fila.close();
            fila=null;
            BaseDeDatabase.close();
            mMap.moveCamera(CameraUpdateFactory.newLatLng(NewPoint));
            mMap.animateCamera(CameraUpdateFactory.newLatLngZoom(NewPoint,18.0f));
        } else {
            Toast.makeText(this,"No existen registros actualmente", Toast.LENGTH_SHORT).show();
            BaseDeDatabase.close();
        }

    }
}
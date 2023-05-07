package com.example.appthesis.activities;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Toast;

import androidx.annotation.NonNull;

import com.example.appthesis.databinding.ActivityMainBinding;
import com.example.appthesis.services.AdminSQLiteOpenHelper;
import com.example.appthesis.services.PermissionService;
import com.example.appthesis.utils.CustomList_Adapter;
import com.example.appthesis.utils.ListView_cultivos;

import java.util.ArrayList;

import javax.inject.Inject;

import dagger.hilt.android.AndroidEntryPoint;
@AndroidEntryPoint
public class MainActivity extends BaseActivity {

    ActivityMainBinding binding;
    @Inject
    PermissionService permissionService;
    AdminSQLiteOpenHelper sqLiteHelper;
    Integer values;

    @Override
    protected void onCreate(Bundle savedInstanceState) {


        super.onCreate(savedInstanceState);


        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.floatPhoto.setOnClickListener(view -> {
                startActivity(new Intent(MainActivity.this, CameraActivity.class));
        });

        binding.floatMap.setOnClickListener(view -> {
                    startActivity(new Intent(MainActivity.this, MapsActivity.class));
                }

        );

        binding.ListViewCult.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
                int valor= values-i;
                Intent startN = new Intent(MainActivity.this, ImagenCacao.class);
                startN.putExtra("dato", String.valueOf(valor));
                startActivity(startN);
            }
        });

    }

    @Override
    protected void onResume() {
        super.onResume();
        Buscar();

        if (!permissionService.ismReadExternalStoragePermissionGranted()) {
            permissionService.getReadExternalStoragePermission(this);
        } else {
            Fill_view();
        }

    }

    public void Buscar(){
        AdminSQLiteOpenHelper admin = new AdminSQLiteOpenHelper(this, "administracion", null, 1);
        SQLiteDatabase BaseDeDatabase = admin.getWritableDatabase();
            Cursor fila = BaseDeDatabase.rawQuery
                    ("select count(0) from CACAO ", null);
            if(fila.moveToFirst()){
                binding.txtRegistros.setText("Registros: "+fila.getString(0));
                values= Integer.valueOf(fila.getString(0));
                BaseDeDatabase.close();
            } else {
                Toast.makeText(this,"No existen registros actualmente", Toast.LENGTH_SHORT).show();
                BaseDeDatabase.close();
            }
        }

        public void Fill_view(){
            AdminSQLiteOpenHelper admin = new AdminSQLiteOpenHelper(this, "administracion", null, 1);
            SQLiteDatabase BaseDeDatabase = admin.getWritableDatabase();
            Cursor fila = BaseDeDatabase.rawQuery
                    ("select name,diagnostico, fecha, image, Id from CACAO ", null);
            ArrayList<ListView_cultivos> infoCultivos = new ArrayList<>();
            if(fila.moveToLast()){
                do {
                    infoCultivos.add(new ListView_cultivos(fila.getString(0),fila.getString(1),fila.getString(2),fila.getBlob(3), fila.getString(4)));
                }while (fila.moveToPrevious());
                fila.close();
                fila=null;
                BaseDeDatabase.close();
            } else {
                Toast.makeText(this,"No existen registros actualmente", Toast.LENGTH_SHORT).show();
                BaseDeDatabase.close();
            }
            CustomList_Adapter adapter = new CustomList_Adapter(this, infoCultivos);
            binding.ListViewCult.setAdapter(adapter);
        }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        permissionService.getReadExternalStoragePermission(this);

        if (permissionService.ismReadExternalStoragePermissionGranted() &&
                requestCode == PermissionService.PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE &&
                grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Fill_view();
        }
    }

}


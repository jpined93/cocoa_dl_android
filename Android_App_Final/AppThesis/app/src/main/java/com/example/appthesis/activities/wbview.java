package com.example.appthesis.activities;

import android.os.Bundle;

import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;

import android.view.View;
import android.webkit.WebViewClient;
import android.widget.Toast;

import androidx.core.view.WindowCompat;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.example.appthesis.databinding.ActivityWbviewBinding;

import com.example.appthesis.R;

public class wbview extends AppCompatActivity {

    private ActivityWbviewBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityWbviewBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        String dato=getIntent().getStringExtra("dato");
        binding.webviewcontainer.setWebViewClient(new WebViewClient());
        binding.webviewcontainer.getSettings().setJavaScriptEnabled(true);
        binding.webviewcontainer.loadUrl("https://drive.google.com/viewerng/viewer?embedded=true&url=" + dato);
        Toast.makeText(this,dato,Toast.LENGTH_SHORT).show();

    }
}
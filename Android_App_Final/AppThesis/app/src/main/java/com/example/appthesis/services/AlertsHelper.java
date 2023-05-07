package com.example.appthesis.services;
import android.content.Context;
import android.util.Log;
import android.view.View;
import android.widget.Toast;
import com.example.appthesis.R;
import com.google.android.material.snackbar.Snackbar;
import javax.inject.Inject;
import dagger.Module;
import dagger.hilt.InstallIn;
import dagger.hilt.android.components.ActivityComponent;
import dagger.hilt.android.qualifiers.ApplicationContext;

@Module
@InstallIn(ActivityComponent.class)
public class AlertsHelper {
    private static final String TAG = AlertsHelper.class.getName();
    private final Context context;

    @Inject
    AlertsHelper(@ApplicationContext Context context) {
        this.context = context;
    }

    public void shortToast(String message) {
        Log.i(TAG, String.format("shortToast: %s", message));
        Toast.makeText(this.context, message, Toast.LENGTH_SHORT).show();
    }

    public void longToast(String message) {
        Log.i(TAG, String.format("longToast: %s", message));
        Toast.makeText(this.context, message, Toast.LENGTH_LONG).show();
    }

    public void shortSimpleSnackbar(View parentView, String message) {
        Log.i(TAG, String.format("shortSimpleSnackBar: %s", message));
        Snackbar.make(parentView, message, Snackbar.LENGTH_SHORT).show();
    }

    public void longSimpleSnackbar(View parentView, String message) {
        Log.i(TAG, String.format("longSimpleSnackBar: %s", message));
        Snackbar.make(parentView, message, Snackbar.LENGTH_LONG).show();
    }
}
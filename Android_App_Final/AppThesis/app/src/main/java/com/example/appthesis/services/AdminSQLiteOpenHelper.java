package com.example.appthesis.services;

import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.database.sqlite.SQLiteStatement;
import androidx.annotation.Nullable;



public class AdminSQLiteOpenHelper extends SQLiteOpenHelper {
    public AdminSQLiteOpenHelper(@Nullable Context context, @Nullable String name, @Nullable SQLiteDatabase.CursorFactory factory, int version) {
        super(context, name, factory, version);
    }

    public void queryData(String sql){
        SQLiteDatabase database = getWritableDatabase();
        database.execSQL(sql);
    }

    public void insertData(String name, String diagnostico, String fecha, String latitud,String longitud, String notas,byte[] image){
        SQLiteDatabase database = getWritableDatabase();
        String sql = "INSERT INTO CACAO VALUES (NULL, ?, ?, ?,?,?,?,?)";
        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();
        statement.bindString(1, name);
        statement.bindString(2, diagnostico);
        statement.bindString(3, fecha);
        statement.bindString(4, latitud);
        statement.bindString(5, longitud);
        statement.bindString(6, notas);
        statement.bindBlob(7, image);
        statement.executeInsert();
        database.close();
    }

    public void updateData( String diagnostico,String id) {
        SQLiteDatabase database = getWritableDatabase();

        String sql = "UPDATE CACAO SET diagnostico = ?  WHERE id = ?";
        SQLiteStatement statement = database.compileStatement(sql);
        statement.bindString(1, diagnostico);
        statement.bindString(2, id);
        statement.execute();
        database.close();
    }

    public  void deleteData(int id) {
        SQLiteDatabase database = getWritableDatabase();

        String sql = "DELETE FROM CACAO WHERE id = ?";
        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();
        statement.bindDouble(1, (double)id);
        statement.execute();
        database.close();
    }

    public Cursor getData(String sql){
        SQLiteDatabase database = getReadableDatabase();
        return database.rawQuery(sql, null);
    }

    @Override
    public void onCreate(SQLiteDatabase sqLiteDatabase) {
        sqLiteDatabase.execSQL("CREATE TABLE IF NOT EXISTS CACAO(Id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "name VARCHAR, diagnostico VARCHAR, fecha VARCHAR, latitud VARCHAR, longitud VARCHAR, notas VARCHAR, image BLOB)");

    }

    @Override
    public void onUpgrade(SQLiteDatabase sqLiteDatabase, int i, int i1) {

    }

}

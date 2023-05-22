# import sys
# sys.path.append('C:\Users\USUARIO\Desktop\Tesis\gitrepo\cocoa_dl_android\FlaskWebService2\src')
# import reader
from unittest import TestCase
from faker import Faker
from app import app

class test(TestCase):
    def setUp(self):
        self.data_factory = Faker()
        self.client = app.test_client()

    def test_health_check(self):
        getRoutes_request = self.client.get("/uImg", headers={'Content-Type': 'application/json'})
        self.assertEqual(getRoutes_request.status_code, 200)
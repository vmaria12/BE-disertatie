"""
Script pentru a crea un superuser cu credențiale admin/admin
"""
import os
import django

# Setează calea către setările Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject1.settings')
django.setup()

from django.contrib.auth.models import User

# Verifică dacă userul admin există deja
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin')
    print('✅ Superuser "admin" a fost creat cu succes!')
    print('   Username: admin')
    print('   Password: admin')
else:
    print('ℹ️  Superuser "admin" există deja.')

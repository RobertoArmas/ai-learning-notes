import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

# Configurar Faker
fake = Faker()
random.seed(42)

# Generar dataset con múltiples transacciones por cliente
num_clients = 50
num_transactions = 5_000
transaction_types = ["crédito", "débito"]
fraud_rate = 0.02  # 2% de transacciones fraudulentas

# Generación de IDs de clientes anonimizados
client_ids = [fake.uuid4() for _ in range(num_clients)]

# Crear historial de transacciones para cada cliente para asegurar múltiples transacciones
client_histories = {client_id: [] for client_id in client_ids}
data = []

for _ in range(num_transactions):
    # Seleccionar un cliente al azar
    client_id = random.choice(client_ids)
    transaction_type = random.choice(transaction_types)
    
    # Generar montos de transacción con alta probabilidad de repetición para simular patrones
    base_amount = round(random.uniform(1, 5000), 2)
    amount = round(base_amount + random.uniform(-10, 10), 2)  # Montos similares
    
    # Generar fecha aleatoria
    date = fake.date_time_between(start_date='-1y', end_date='now')
    location = fake.city()
    
    # Crear historial para el cliente (simulando transacciones similares sin ser marcadas como fraude)
    history = client_histories[client_id]
    history.append(amount)
    
    # Etiqueta de fraude con una probabilidad de 2%
    is_fraud = 1 if random.random() < fraud_rate else 0
    
    # Añadir transacción al dataset
    data.append([client_id, amount, location, transaction_type, date, history.copy(), is_fraud])

# Crear DataFrame
df_refined = pd.DataFrame(data, columns=[
    "Client_ID", "Amount", "Location", "Card_Type", "Transaction_Date", "Transaction_History", "Fraud_Label"
])

df_refined["Amount"] = df_refined.apply(
    lambda row: round(random.uniform(4000, 5000), 2) if row["Fraud_Label"] == 1 else row["Amount"], axis=1
)

df_refined["Merchant_Name"] = [fake.company() for _ in range(len(df_refined))]  # Nombre del comerciante
df_refined["Merchant_Category"] = [random.choice(["Restaurantes", "Tecnología", "Ropa", "Alimentos", "Joyería", "Electrónica", "Combustible", "Viajes", "Entretenimiento", "Supermercados"]) for _ in range(len(df_refined))]
df_refined["Terminal_ID"] = [fake.bothify(text="TERM-####-???") for _ in range(len(df_refined))]  # ID del terminal de pago
df_refined["Device_Type"] = [random.choice(["Móvil", "POS Fijo", "Online", "Cajero Automático"]) for _ in range(len(df_refined))]
df_refined["Payment_Method"] = [random.choice(["Tarjeta Física", "Contacto", "NFC", "QR", "Pago Online"]) for _ in range(len(df_refined))]
df_refined["Currency"] = [random.choice(["USD"]) for _ in range(len(df_refined))]
df_refined["Customer_Segment"] = [random.choice(["Retail", "Corporativo", "Pequeña Empresa", "Gobierno"]) for _ in range(len(df_refined))]

print(df_refined.head())

# Exportar el dataset refinado a un archivo CSV
csv_file_path = "output/Dataset_Transacciones_desquilibrado.csv"
df_refined.to_csv(csv_file_path, index=False, sep=";")

# Devolver el enlace de descarga al usuario
csv_file_path
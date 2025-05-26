import subprocess
import time
import os

print("✅ Εκκίνηση εκπαίδευσης (main.py)...")
subprocess.run(["python", "main.py"])

print("✅ Εκπαίδευση ολοκληρώθηκε. Πίνακες αποθηκεύτηκαν.")

print("🎮 Είσαι έτοιμος για προσομοίωση.")
print("➡ Πάτησε ENTER για να ξεκινήσει το pygame_simulation.py")
input()

# Τρέχει το pygame_simulation ξεχωριστά σε νέο παράθυρο
subprocess.Popen(["python", "pygame_simulation.py"])

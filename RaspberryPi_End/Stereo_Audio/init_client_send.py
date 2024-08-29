import socket
from recordAUDIO import recorDAUDIO

def run_client():
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_ip = '192.168.0.23' #'192.168.1.30'
    server_port = 2222
    client.connect((server_ip, server_port))
    
    while True:
        msg = input('Enter message: ')
        
        if msg.lower() == 'trial':
            msg = '@#@_0.678_Jogging'
        
        if msg.lower() == 'start':
            msg = 'start'
            client.send(msg.encode('utf-8')[:1024])
            rec_auds = recorDAUDIO()
            rec_auds.start_recording_now()
          
        client.send(msg.encode('utf-8')[:1024])
        
        if msg.lower() == 'closed':
            break
        
    client.close()
    print('Connection to server closed')

run_client()

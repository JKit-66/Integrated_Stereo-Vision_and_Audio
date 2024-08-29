import socket
import time
from threading import Timer
from picam_take_vid import take_Vids

bufferSize = 1024
msg = 'HappyClient'.encode('utf-8')
ServerPort=2222
ServerIP = '192.168.0.23' #'192.168.1.30'        #'10.9.0.188'
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((ServerIP, ServerPort))
print('Server now listening...')

cnt = 0
while True:
    messg, addr = s.recvfrom(bufferSize)
    messg=messg.decode('utf-8')
    print(messg)
    #print('Client address: ', addr[0])
    if messg.lower() =='+' or messg.lower() =='=':
        cnt = cnt + 5
        print(cnt)
        
    elif messg.lower() =='-':
        cnt = cnt - 5
        print(cnt)
        
    elif 'start' in messg.lower():
        print(messg)
        viDs = take_Vids()
        time.sleep(2.3)
        viDs.final_rec_vids()
        
    elif '@#@' in messg.lower():
        uncoder_ = messg.split('_')
        conf_scr = float(uncoder_[1])
        clss_det = uncoder_[2]
        
        print(conf_scr, clss_det)
    
    
    #msg=str(cnt)
    #msg=msg.encode('utf-8')
    #s.sendto(msg,addr)

#s.listen(5)

'''def bg_cont():
    msg = 'Hello Client'
    clientsocket.send(bytes(msg, "utf-8"))
    Timer(5, bg_cont).start()

while True:
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established.")
    bg_cont()
    
'''

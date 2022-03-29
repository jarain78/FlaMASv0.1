import os
# Config XMPP Server
xmpp_server = '192.168.1.3'

web_port = 3000
url = "localhost"
jid_domain = "@" + xmpp_server

# FSM name states Central Federado
SETUP_STATE_CF = "SETUP_STATE"
RECEIVE_STATE_CF = "RECEIVE_STATE"
MEAN_STATE_CF = "MEAN_STATE"
SEND_STATE_CF = "SEND_STATE"

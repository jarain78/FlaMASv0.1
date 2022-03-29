import os
# Config XMPP Server
xmpp_server = '192.168.1.3'

web_port = 3000
url = "localhost"
jid_domain = "@" + xmpp_server

# FSM name states Central Federado
SETUP_STATE_AG = "SETUP_STATE"
RECEIVE_STATE_AG = "RECEIVE_STATE"
TRAIN_STATE_AG = "TRAIN_STATE"
SEND_STATE_AG = "SEND_STATE"

agent_to_subscribe = "central_federado" + jid_domain
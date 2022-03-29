from termcolor import colored

import Config
import time
import pickle
import codecs
from spade.agent import Agent
from spade.message import Message
from timeit import default_timer as timer
from spade.behaviour import FSMBehaviour, State, OneShotBehaviour
from FederatedLearning4MultiAgentSystems.PytorchFederateLearning.Utilities import Utilities


# ======================================================================================================================
# This class is in charge of managing the presence of all Agents
# ======================================================================================================================

class PresenceBehav(OneShotBehaviour):
    def on_available(self, jid, stanza):
        print(
            "[{}] Agent {} is available.".format(self.agent.name, jid.split("@")[0])
        )

    def on_subscribed(self, jid):
        print(
            "[{}] Agent {} has accepted the subscription.".format(
                self.agent.name, jid.split("@")[0]
            )
        )
        print(
            "[{}] Contacts List: {}".format(
                self.agent.name, self.agent.presence.get_contacts()
            )
        )

    def on_subscribe(self, jid):
        print(
            "[{}] Agent {} asked for subscription. Let's aprove it.".format(
                self.agent.name, jid.split("@")[0]
            )
        )
        self.presence.approve(jid)
        self.presence.subscribe(jid)

    async def run(self):
        self.presence.set_available()
        self.presence.on_subscribe = self.on_subscribe
        self.presence.on_subscribed = self.on_subscribed
        self.presence.on_available = self.on_available


# ======================================================================================================================
# It is the state machine of the federated central agent, to which all the agents will connect
# and calculate the average of the weights sent by each of them.
# ======================================================================================================================

class CentralFederatedAgent(FSMBehaviour):
    async def on_start(self):
        print(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        print(f"FSM finished at state {self.current_state}")
        await self.agent.stop()


class setup_state(State):
    async def run(self):
        print("-    This is the setup state")
        self.set_next_state(Config.RECEIVE_STATE_CF)


class receive_state(State):

    def __init__(self):
        super().__init__()
        self.agents_name = []
        self.old_agent = ""
        self.local_weights = []
        self.local_losses = []

    def check_if_all_ag_send_msg(self, msg):
        agent_sender = msg.sender
        body_msg = msg.body
        meta_msg = msg.metadata

        self.agent.avail_contacts = {key: None for key, value in self.agent.presence.get_contacts().items()
                                     if 'presence' in value and value['presence'].show is not None}
        n_agents_in_presence = len(self.agent.avail_contacts)

        print(n_agents_in_presence)
        '''if n_agents_in_presence != 0:
            print(colored('=' * 30, 'green'))
            print(colored("N-Contacts: ", n_agents_in_presence, 'cyan'))
            print(colored("Body-MSG: ", body_msg, 'cyan'))
            print(colored("MSG Metadata: ", meta_msg, 'cyan'))
            print(colored('=' * 30, 'green'))'''

        self.set('n-agents', n_agents_in_presence)
        self.set('body_msg', body_msg)
        self.set('msg_metadata', meta_msg)

        # check if all agents sent it the message
        for key in self.agent.avail_contacts:
            if str(key) in str(agent_sender):
                print(str(agent_sender) + " " + str(key))
                self.agents_name.append(str(key))

        if str(msg.sender) != self.old_agent:
            print('msg sender: ', msg.sender)
            msg_data = msg.body.split(':""')[1].split('"}')[0]
            weights_and_losses = msg_data.split('|')
            self.local_weights.append(weights_and_losses[0])
            self.local_losses.append(weights_and_losses[1])

        self.old_agent = str(msg.sender)

        if n_agents_in_presence == len(self.agents_name):
            self.set('agents_names', self.agents_name)
            return True
        else:
            return False

    async def run(self):
        # print("-    This is the receive state")
        msg = await self.receive(timeout=None)

        if msg:
            all_agents_sent_messages = self.check_if_all_ag_send_msg(msg)
            if all_agents_sent_messages:
                self.agents_name = []
                self.set_next_state(Config.MEAN_STATE_CF)
        else:
            self.set('list_local_weights', self.local_weights)
            self.set('list_local_loss', self.local_losses)

            self.set_next_state(Config.RECEIVE_STATE_CF)


class mean_state(State):

    def __init__(self):
        super().__init__()
        self.utilities = Utilities()

    async def run(self):
        start_time_mean = timer()
        print("-    This is the mean state")
        loss_avg = 0

        value_msg = self.get('body_msg')
        meta_msg = self.get('msg_metadata')
        conversation_type = dict(meta_msg)

        local_weights = self.get('list_local_weights')
        local_losses = self.get('list_local_loss')

        if 'federatedLearning' in conversation_type['conversation']:
            # unpickled_local_weights = pickle.loads(codecs.decode(local_weights.encode(), "base64"))
            avg_local_weights = self.utilities.average_weights(local_weights)

            for j in local_losses:
                unpickled_local_losses = pickle.loads(codecs.decode(j.encode(), "base64"))
                loss_avg = sum(unpickled_local_losses) / len(unpickled_local_losses)

            serial_avg_local_weights = codecs.encode(pickle.dumps(avg_local_weights), "base64").decode()
            serial_avg_local_losses = codecs.encode(pickle.dumps(loss_avg), "base64").decode()

            self.set('avg_local_weights', serial_avg_local_weights)
            self.set('avg_local_loss', serial_avg_local_losses)

            '''print(colored('=' * 30, 'green'))
            print(avg_local_weights)
            print(colored('=' * 30, 'green'))
            print(loss_avg)'''

        end_time_mean = timer()
        print(colored('=' * 30, 'green'))
        delta_t = end_time_mean - start_time_mean
        print(colored(delta_t, 'red'))
        print(colored('=' * 30, 'green'))
        self.set_next_state(Config.SEND_STATE_CF)


class send_state(State):
    async def run(self):
        print("-    This is the sender state")
        agents_name = self.get('agents_names')

        for agents2send_msg in agents_name:
            msg = Message(to=str(agents2send_msg))
            serial_avg_local_weights = self.get('avg_local_weights')
            serial_avg_local_losses = self.get('avg_local_loss')

            msg.body = '{' + '"' + "value:" + '"' + '"' + str(serial_avg_local_weights) + "|" + str(
                serial_avg_local_losses) + '"' + '}'

            await self.send(msg)

        self.set_next_state(Config.RECEIVE_STATE_CF)


class FSMAgent(Agent):
    async def setup(self):
        fsm = CentralFederatedAgent()
        fsm.add_state(name=Config.SETUP_STATE_CF, state=setup_state(), initial=True)
        fsm.add_state(name=Config.RECEIVE_STATE_CF, state=receive_state())
        fsm.add_state(name=Config.MEAN_STATE_CF, state=mean_state())
        fsm.add_state(name=Config.SEND_STATE_CF, state=send_state())

        fsm.add_transition(source=Config.SETUP_STATE_CF, dest=Config.RECEIVE_STATE_CF)
        fsm.add_transition(source=Config.RECEIVE_STATE_CF, dest=Config.MEAN_STATE_CF)
        fsm.add_transition(source=Config.MEAN_STATE_CF, dest=Config.SEND_STATE_CF)
        fsm.add_transition(source=Config.SEND_STATE_CF, dest=Config.RECEIVE_STATE_CF)

        fsm.add_transition(source=Config.RECEIVE_STATE_CF, dest=Config.RECEIVE_STATE_CF)
        self.add_behaviour(PresenceBehav())
        self.add_behaviour(fsm)


if __name__ == "__main__":

    jid_launch = "central_federado" + Config.jid_domain
    passwd_launch = "test"

    fsmagent = FSMAgent(jid_launch, passwd_launch)
    future = fsmagent.start()
    future.result()

    while fsmagent.is_alive():
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            fsmagent.stop()
            break
    print("Agent finished")

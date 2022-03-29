import time
import codecs
import pickle
import Config
from termcolor import colored
from spade.agent import Agent
from spade.message import Message
from spade.behaviour import FSMBehaviour, State, OneShotBehaviour, PeriodicBehaviour
from FederatedLearning4MultiAgentSystems.PytorchFederateLearning.Federated import Federated

federated_learning = Federated()
federated_learning.build_model()
print(colored('=' * 30, 'blue'))
federated_learning.print_model()
print(colored('=' * 30, 'blue'))
federated_learning.set_model()


class PresenceBehav(OneShotBehaviour):

    def on_available(self, jid, stanza):
        print("[{}] Agent {} is available.".format(self.agent.name, jid.split("@")[0]))

    def on_subscribed(self, jid):
        print("[{}] Agent {} has accepted the subscription.".format(self.agent.name, jid.split("@")[0]))
        print("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()))

    def on_subscribe(self, jid):
        print("[{}] Agent {} asked for subscription. Let's aprove it.".format(self.agent.name, jid.split("@")[0]))
        self.presence.approve(jid)
        self.presence.subscribe(jid)

    async def run(self):
        self.presence.on_subscribe = self.on_subscribe
        self.presence.on_subscribed = self.on_subscribed
        self.presence.on_available = self.on_available

        self.presence.set_available()
        self.presence.subscribe(Config.agent_to_subscribe)


class PresencePeriodicBehav(PeriodicBehaviour):

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


class Agent_3(FSMBehaviour):

    async def on_start(self):
        print(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        print(f"FSM finished at state {self.current_state}")
        await self.agent.stop()


class setup_state(State):
    async def run(self):
        print("-    This is the setup state")
        self.set_next_state(Config.TRAIN_STATE_AG)


class receive_state(State):
    async def run(self):
        print("-    This is the receive state")
        print("-    Waiting for Centrala Agent message")
        msg = await self.receive(timeout=1)
        if msg:
            if "central_federado" in msg.sender:
                msg_data = msg.body.split(':""')[1].split('"}')[0]
                weights_and_losses = msg_data.split('|')

                unpickled_local_weight = pickle.loads(codecs.decode(weights_and_losses[0].encode(), "base64"))
                unpickled_local_losses = pickle.loads(codecs.decode(weights_and_losses[1].encode(), "base64"))
                # print(unpickled_local_weight)
                # print(unpickled_local_losses)
                federated_learning.add_new_local_weight_local_losses(unpickled_local_weight, unpickled_local_losses)

                print(colored('=' * 30, 'blue'))
                federated_learning.set_model()

                self.set_next_state(Config.TRAIN_STATE_AG)
        else:
            print("-    Change State")
            self.set_next_state(Config.RECEIVE_STATE_AG)


class train_state(State):

    def deep_learning(self):
        federated_learning.train_global_model()

        # Deep Learning Local
        # --------------------------------------------------------------------------------------------------------------
        print(colored('=' * 30, 'green'))
        print("- Update State")

        print("    [", self.agent.name, "] --- Trained your Local Model")
        print(colored('=' * 30, 'green'))

        local_weights, local_losses = federated_learning.train_local_model(AgName="pepita_hp_3", epoch=1)

        #federated_learning.average_all_weights(local_weights, local_losses)
        #federated_learning.get_acc()

        serial_local_weights = codecs.encode(pickle.dumps(local_weights), "base64").decode()
        serial_local_losses = codecs.encode(pickle.dumps(local_losses), "base64").decode()

        print(" --- Setting local weights and local losses")

        # print(colored('=' * 50, 'red'))
        # print("SET WEIGHTS")
        # print(serial_local_weights)

        self.set('local_weights', serial_local_weights)
        self.set('local_losses', serial_local_losses)

        # ---------------------------------------------------------------------------------------------------------------

    async def run(self):
        print("-    This is the training state")
        self.deep_learning()
        self.set_next_state(Config.SEND_STATE_AG)


class send_state(State):

    def send_message(self, recipient):
        msg = Message(to=recipient)  # Instantiate the message

        local_weights0 = self.get('local_weights')
        local_losses0 = self.get('local_losses')

        if local_weights0 == None and local_losses0 == None:
            msg.body = 'No Data'
            msg.set_metadata('conversation', 'federatedLearning')
            return msg
        else:
            print("    Send Message to ", recipient)
            # https://stackoverflow.com/questions/30469575/how-to-pickle-and-unpickle-to-portable-string-in-python-3
            # msg.body = '{' + '"' + "local_weights:" + '"' + local_weights0 + '"' + "," + "value1" + ":" + '"' + local_losses0 + '"' + '}'
            msg.body = '{' + '"' + "value:" + '"' + '"' + str(local_weights0) + "|" + str(
                local_losses0) + '"' + '}'

            msg.set_metadata('conversation', 'federatedLearning')
            return msg

    async def run(self):
        print("-    This is the sender state")

        recipient = "central_federado" + "@" + Config.xmpp_server
        msg = self.send_message(recipient)
        await self.send(msg)
        self.set_next_state(Config.RECEIVE_STATE_AG)


class FSMAgent(Agent):
    async def setup(self):
        fsm = Agent_3()
        fsm.add_state(name=Config.SETUP_STATE_AG, state=setup_state(), initial=True)
        fsm.add_state(name=Config.RECEIVE_STATE_AG, state=receive_state())
        fsm.add_state(name=Config.TRAIN_STATE_AG, state=train_state())
        fsm.add_state(name=Config.SEND_STATE_AG, state=send_state())

        fsm.add_transition(source=Config.SETUP_STATE_AG, dest=Config.TRAIN_STATE_AG)
        fsm.add_transition(source=Config.TRAIN_STATE_AG, dest=Config.SEND_STATE_AG)
        fsm.add_transition(source=Config.SEND_STATE_AG, dest=Config.RECEIVE_STATE_AG)
        fsm.add_transition(source=Config.RECEIVE_STATE_AG, dest=Config.RECEIVE_STATE_AG)
        fsm.add_transition(source=Config.RECEIVE_STATE_AG, dest=Config.TRAIN_STATE_AG)

        # fsm.add_transition(source=Config.SETUP_STATE_AG, dest=Config.RECEIVE_STATE_AG)

        self.add_behaviour(fsm)
        self.add_behaviour(PresenceBehav())


if __name__ == "__main__":

    jid_launch = "pepita_hp_3" + "@" + Config.xmpp_server
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

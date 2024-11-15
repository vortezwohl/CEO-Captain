import logging

from langchain_core.language_models import BaseChatModel

from ceo import Agent
from ceo.prompt.prompt import Prompt

log = logging.getLogger('ceo.prompt')


class DispatcherPrompt(Prompt):
    def __init__(self, query: str, crews: list[Agent], ext_context: str = ''):
        self.crews = crews
        prompt = dict()
        for crew in self.crews:
            prompt[crew.name] = crew.introduce()
        prompt = ('Precondition: Below are the crews you can call for help (you can only call the following crews). '
                  f'Now there is a user query: "{query}"\n'
                  'Task: What you need to do is to plan your workflow with your crews based on user query.\n'
                  '(User query might contains many steps, '
                  'think carefully about every step and plan your workflow with your crews)\n'
                  "(A crew member can only be called once)\n"
                  "(Sometimes some of the crews are irrelevant to user's query. Make sure to call your crews properly.)\n"
                  'Output format: [{crew_member_1.name}, {crew_member_2.name}, ...] sequential and well-organized with no additional redundant information\n'
                  'Example output: [task_manager, bus_driver, song_writer]\n'
                  f'Your crews: {prompt}\n')
        super().__init__(prompt, ext_context)
        log.debug(f'DispatcherPrompt: {self.prompt}')

    def invoke(self, model: BaseChatModel) -> list[Agent]:
        results = model.invoke(self.prompt)
        results = results.content[1:-1].split(',')
        _fin_results = list()
        for _a_result in results:
            for crew in self.crews:
                if (crew not in _fin_results
                        and crew.name == _a_result.strip()):
                    _fin_results.append(crew)
        return _fin_results

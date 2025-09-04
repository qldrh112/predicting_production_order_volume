from jupyter_client.manager import KernelManager

class JupyterKernel:
    def __init__(self):
        self.km = KernelManager()
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()

    def run_code(self, code: str) -> str:
        """코드 실행 후 출력 결과 반환"""
        self.kc.execute(code)
        output = ""
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=2)
                if msg['msg_type'] == 'stream':
                    output += msg['content'].get('text', '')
                elif msg['msg_type'] == 'execute_result':
                    output += str(msg['content']['data'].get('text/plain', ''))
            except Exception:
                break
        return output

    def shutdown(self):
        self.kc.stop_channels()
        self.km.shutdown_kernel()

from jupyter_client.manager import KernelManager

class JupyterKernel:
    def __init__(self):
        self.km = KernelManager()
        self.km.start_kernel()
        self.kernel_client = self.km.client()
        self.kernel_client.start_channels()

    def execute_code(self, code: str) -> str:
        """Python 코드를 실행하고 결과를 반환하는 함수"""
        self.kernel_client.execute(code)
        output = ""
        while True:
            try:
                msg = self.kernel_client.get_iopub_msg(timeout=10)
                if msg['msg_type'] == 'stream':
                    output += msg['content'].get('text', '')
                elif msg['msg_type'] == 'execute_result':
                    output += str(msg['content']['data'].get('text/plain', ''))
            except Exception:
                break
        return output

    def get_variable(self, name: str) -> str:
        """특정 변수의 값을 조회하는 함"""
        code = f"print({name})"
        return self.execute_code(code)

    def shutdown(self):
        """Jupyter kernel 클라이언트를 종료하고, kernel manager를 닫는 함수"""
        self.kernel_client.stop_channels()
        self.km.shutdown_kernel()
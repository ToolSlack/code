"""Guard-workload phase: measured cost and native residency gates."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    if not os.environ.get('TOOLSLACK_RUNTIME_ROOT'):
        raise RuntimeError('An independently restoring GPU guard is required')
    profile = os.environ['TOOLSLACK_SERVICE_PROFILE']
    common = ['--proxy', os.environ['TOOLSLACK_PROXY_URL'],
              '--upstream', os.environ['TOOLSLACK_ENGINE_URL'],
              '--service-profile', profile]
    output = Path(os.environ['TOOLSLACK_COST_PROFILE'])
    subprocess.run([sys.executable, '-m', 'calibration.prefix', *common,
                    '--output', str(output)], check=True)
    subprocess.run([sys.executable, '-m', 'calibration.lifecycle', *common,
                    '--output', str(output.with_name('KV_LIFECYCLE_GATE.json'))], check=True)


if __name__ == '__main__':
    main()

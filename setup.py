import subprocess


def create_virtualenv(env_name):
    subprocess.run(['python3', '-m', 'venv', env_name])


def pip_install(env_name, requirements_file):
    subprocess.run([f'{env_name}/bin/pip3', 'install', '-r', requirements_file])


if __name__ == "__main__":
    env_name = 'fake_news'
    create_virtualenv(env_name)
    requirements_file = 'requirements.txt'
    pip_install(env_name, requirements_file)
function exit_err
    echo $argv
    exit
end

if not command -s pipenv > /dev/null
    echo 'Installing pipenv...'
    if not pip3 install pipenv -U --user
        exit_err 'Failed to install pipenv'
    end
end

if not set cuda (python3 -c 'import torch; print(torch.version.cuda)' ^/dev/null) \
    or not string match '10' $cuda
    exit_err 'PyTorch is not installed or not for CUDA 10'
end

if not pipenv --venv >/dev/null 2>&1
    echo 'Creating pipenv virtual enviroment...'
    if not pipenv --three --site-packages
        exit_err 'Failed to create virtualenv'
    end
end

echo 'Run pipenv install'
pipenv install
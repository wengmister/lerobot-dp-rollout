ZKW, FRANKAXHAND DEPLOYMENET

    git submodule --init --recursive

    uv pip install -e .
    
    cd dex_retargeting 
    uv pip install -e .
    
    cd .. 

    cd franka_xhand_teleoperator
    uv pip install -e .
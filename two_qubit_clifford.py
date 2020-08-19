def two_qubit_clifford(generators_q1, generators_q2, plus_op_parallel, two_qubit_gate=None, two_qubit_gate_name='CZ', error=1e-3):
    # see https://arxiv.org/pdf/1210.7011.pdf
    c_q1 = generate_group(generators_q1)
    c_q2 = generate_group(generators_q2)
    s_q1 = {}
    s_q2 = {}
    pi2_q1 = {}
    pi2_q2 = {}
    # finding s-gates
    def find_s_gates_from_cliffords(cliffords):
        for name, clifford in cliffords.items():
            # has no 2-nd-order axis
            if not np.abs(np.abs(np.trace(clifford['unitary'] @ clifford['unitary'])) - clifford['unitary'].shape[0]) < error:
                # has a 3-rd order axis
                if np.abs(np.abs(np.trace(clifford['unitary'] @ clifford['unitary'] @ clifford['unitary'])) - clifford['unitary'].shape[0]) < error:
                    generator = clifford
        s = {}
        s0 = generator['unitary'] @ generator['unitary'] @ generator['unitary']
        s1 = generator['unitary']
        s2 = generator['unitary'] @ generator['unitary']
        norm = np.abs(np.trace(s0))
        for name, clifford in cliffords.items():
            if np.abs(np.abs(np.sum(s0 * np.conj(clifford['unitary']))) - norm)<error:
                s[name] = clifford
            if np.abs(np.abs(np.sum(s1 * np.conj(clifford['unitary']))) - norm) < error:
                s[name] = clifford
            if np.abs(np.abs(np.sum(s2 * np.conj(clifford['unitary']))) - norm) < error:
                s[name] = clifford
        return s
    s_q1 = find_s_gates_from_cliffords(c_q1)
    s_q2 = find_s_gates_from_cliffords(c_q2)
    # finding pi/2-gates
    for name, clifford in c_q1.items():
        square = clifford['unitary'] @ clifford['unitary']
        if np.abs(np.trace(square)) - clifford['unitary'].shape[0] > 1e-3: # not a pauli gate
            if np.abs(np.trace(square @ square)) - clifford['unitary'].shape[0] < 1e-3: # 4-fold symmetry
                pi2_q1[name] = clifford

    for name, clifford in c_q2.items():
        square = clifford['unitary'] @ clifford['unitary']
        if np.abs(np.trace(square)) - clifford['unitary'].shape[0] > 1e-3:  # not a pauli gate
            if np.abs(np.trace(square @ square)) - clifford['unitary'].shape[0] < 1e-3:  # 4-fold symmetry
                pi2_q2[name] = clifford


    group = {}
    #tensor product
    for name1, clifford1 in c_q1.items():
        for name2, clifford2 in c_q2.items():
            group[name1+' '+name2] = {'unitary':clifford1['unitary']@clifford2['unitary'],
                                      'pulses':plus_op_parallel(clifford1['pulses'],clifford2['pulses'])}

    # simultaneous single_qubit_clifford group
    if two_qubit_gate is None:
        return group
    # two_qubit_clifford group
    #two types of gates - cz and iswap, something else? bswap for future
    if two_qubit_gate_name is "CZ":
		# https://media.nature.com/original/nature-assets/nature/journal/v508/n7497/extref/nature13171-s1.pdf
        #single qubit gates decomposition
        ry= lambda x: np.asarray([[np.cos(x/2),-np.sin(x/2)],[np.sin(x/2),np.cos(x/2)]])
        rx= lambda x: np.asarray([[np.cos(x/2),-1j*np.sin(x/2)],[-1j*np.sin(x/2),np.cos(x/2)]])
        unit=np.asarray([[1,0],[0,1]])
        names1=["Y/2_1","-Y/2_1"]
        names2=["Y/2_2","-Y/2_2","X/2_2","-X/2_2"]
        gates_for_1_qubit=dict()
        gates_for_2_qubit=dict()
        for i,j in zip([ry(np.pi/2),ry(-np.pi/2)],names1):
            found = False
            for name1, clifford1 in c_q1.items():
                if np.abs(np.sum(np.kron(i,unit)*np.conj(clifford1['unitary']))) > 4-error:
                    found=True
                    gates_for_1_qubit[j]=clifford1
                    break
        for i,j in zip([ry(np.pi/2),ry(-np.pi/2),rx(np.pi/2),rx(-np.pi/2)],names2):
            found = False
            for name2, clifford2 in c_q2.items():
                if np.abs(np.sum(np.kron(unit,i)*np.conj(clifford2['unitary']))) > 4-error:
                    found=True
                    gates_for_2_qubit[j]=clifford2
                    break

        #cnot
        cnot_name = "-Y/2_2" + ' ' + two_qubit_gate_name + ' ' + "Y/2_2"
        cnot = {'pulses': gates_for_2_qubit["Y/2_2"]['pulses']+two_qubit_gate['pulses']+gates_for_2_qubit["-Y/2_2"]['pulses'],
                'unitary': np.asarray([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,0,1],
                                       [0,0,1,0],])}
        #iswap
        iswap_name = "-Y/2_1" + ' ' + "-X/2_2" + ' ' + two_qubit_gate_name + ' ' +"Y/2_1" + ' ' + "-X/2_2" + ' ' + two_qubit_gate_name + ' ' + "Y/2_1" + ' ' + "X/2_2"
        iswap = {'pulses': gates_for_1_qubit["Y/2_1"]['pulses']+gates_for_2_qubit["X/2_2"]['pulses']+two_qubit_gate['pulses']+
                           gates_for_1_qubit["Y/2_1"]['pulses']+gates_for_2_qubit["-X/2_2"]['pulses']+two_qubit_gate['pulses']+
                           gates_for_1_qubit["-Y/2_1"]['pulses']+gates_for_2_qubit["-X/2_2"]['pulses'],
                'unitary': np.asarray([[1,0,0,0],
                                     [0,0,1j,0],
                                     [0,1j,0,0],
                                     [0,0,0,1]])}
        #swap
        swap_name = "-Y/2_2" + ' ' + two_qubit_gate_name +"-Y/2_1" + ' ' + "Y/2_2" + ' ' + two_qubit_gate_name + ' ' + "Y/2_1" + ' ' + "-Y/2_2"+ ' ' + two_qubit_gate_name + ' ' + "Y/2_2"
        swap = {'pulses': gates_for_2_qubit["Y/2_2"]['pulses'] +two_qubit_gate['pulses']+
                           gates_for_1_qubit["Y/2_1"]['pulses']+gates_for_2_qubit["-Y/2_2"]['pulses']+two_qubit_gate['pulses']+
                           gates_for_1_qubit["-Y/2_1"]['pulses']+gates_for_2_qubit["Y/2_2"]['pulses']+two_qubit_gate['pulses']+
                           gates_for_2_qubit["-Y/2_2"]['pulses'],
              'unitary': np.asarray([[1,0,0,0],
                                     [0,0,1,0],
                                     [0,1,0,0],
                                     [0,0,0,1]])}

    if two_qubit_gate_name is "iSWAP":
		# https://journals.aps.org/pra/pdf/10.1103/PhysRevA.67.032301
		# works only with the help of Z-gates
         #single qubit gates decomposition
        rz= lambda x: np.asarray([[np.exp(-1j/2*x),0],[0,np.exp(1j/2*x)]])
        rx= lambda x: np.asarray([[np.cos(x/2),-1j*np.sin(x/2)],[-1j*np.sin(x/2),np.cos(x/2)]])
        unit=np.asarray([[1,0],[0,1]])
        names1=["-Z/2_1","X/2_1","-X/2_1"]
        names2=["X/2_2","Z/2_2","-X/2_2"]
        gates_for_1_qubit=dict()
        gates_for_2_qubit=dict()
        for i,j in zip([rz(-np.pi/2),rx(np.pi/2),rx(-np.pi/2)],names1):
            found = False
            for name1, clifford1 in c_q1.items():
                if np.abs(np.sum(np.kron(i,unit)*np.conj(clifford1['unitary']))) > 4-error:
                    found=True
                    gates_for_1_qubit[j]=clifford1
                    break
        for i,j in zip([rx(np.pi/2),rz(np.pi/2),rx(-np.pi/2)],names2):
            found = False
            for name2, clifford2 in c_q2.items():
                if np.abs(np.sum(np.kron(unit,i)*np.conj(clifford2['unitary']))) > 4-error:
                    found=True
                    gates_for_2_qubit[j]=clifford2
                    break

        #cnot
        cnot_name = "-Z/2_2" + ' '+ "-Z/2_1"+ ' '+ "-X/2_2"+ two_qubit_gate_name + ' ' + "X/2_1"+ ' ' + two_qubit_gate_name + ' ' + "Z/2_2"
        cnot = {'pulses': gates_for_2_qubit["Z/2_2"]['pulses']+two_qubit_gate['pulses']+gates_for_1_qubit["X/2_1"]['pulses']+
						 two_qubit_gate['pulses']+gates_for_2_qubit["-X/2_2"]['pulses']+gates_for_1_qubit["-Z/2_1"]['pulses']+gates_for_2_qubit["Z/2_2"]['pulses'],
                'unitary': np.asarray([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,0,1],
                                       [0,0,1,0],])}
        #iswap
        iswap_name =  two_qubit_gate_name
        iswap = {'pulses': two_qubit_gate['pulses'],
                'unitary': np.asarray([[1,0,0,0],
                                     [0,0,1j,0],
                                     [0,1j,0,0],
                                     [0,0,0,1]])}
        #swap
        swap_name = two_qubit_gate_name + ' ' +"-X/2_2"  + two_qubit_gate_name + ' ' + "-X/2_1" + ' ' + two_qubit_gate_name +  ' ' +"-X/2_2"
        swap = {'pulses': gates_for_2_qubit["-X/2_2"]['pulses'] +two_qubit_gate['pulses']+
                           gates_for_1_qubit["-X/2_1"]['pulses']+two_qubit_gate['pulses']+
                           gates_for_2_qubit["-X/2_2"]['pulses']+two_qubit_gate['pulses'],
              'unitary': np.asarray([[1,0,0,0],
                                     [0,0,1,0],
                                     [0,1,0,0],
                                     [0,0,0,1]])}
    #common for two-qubit gates
    # cnot-like class
    for name1, clifford1 in c_q1.items():
        for name2, clifford2 in c_q2.items():
            for name3, s1 in s_q1.items():
                for name4, s2 in s_q2.items():
                    #print (len(group), name1+' '+name2+' '+cphase_name+' '+name3+' '+name4)
                    group[name1+' '+name2+' '+cnot_name+' '+name3+' '+name4] = {
                        'unitary': clifford1['unitary']@clifford2['unitary']@cnot['unitary']@s1['unitary']@s2['unitary'],
                        'pulses': "cnot"}
    # iswap-like class
    for name1, clifford1 in c_q1.items():
        for name2, clifford2 in c_q2.items():
            for name3, s1 in s_q1.items():
                for name4, s2 in s_q2.items():
                    group[name1+' '+name2+' '+iswap_name+' '+name3+' '+name4] = {
                        'unitary': clifford1['unitary']@clifford2['unitary']@iswap['unitary']@s1['unitary']@s2['unitary'],
                        'pulses': "iSWAP"}
    # swap-like class
    for name1, clifford1 in c_q1.items():
        for name2, clifford2 in c_q2.items():
            group[name1+' '+name2+' '+swap_name] = {'unitary':clifford1['unitary']@clifford2['unitary']@swap['unitary'],
                                          'pulses':"SWAP"}
    return group

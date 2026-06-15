# Prezentare: Model Non-LLM - AI care învață singur Snake prin Reinforcement Learning

## Introducere
„Bună ziua! Astăzi vă prezint un proiect bazat pe un model de **Inteligență Artificială Non-LLM**. Când spunem AI astăzi, majoritatea oamenilor se gândesc la ChatGPT sau alte LLM-uri (Large Language Models) care generează text predictiv. Modelul meu este complet diferit. Acesta nu procesează limbaj și nu a fost antrenat pe texte de pe internet. În schimb, este un agent care învață interacționând cu un mediu — jocul clasic de Snake — printr-un proces numit **Reinforcement Learning (Învățare prin Recompensă)**.”

## De ce este „Wow”?
„Partea fascinantă a acestui proiect este să vedeți evoluția agentului în timp real.
La început (în primele zeci de episoade), agentul este complet „orb” și acționează la întâmplare. Se lovește constant de pereți, se învârte în cerc și rareori nimerește mâncarea. Puteți vedea asta rulând agentul random pe care l-am creat ca baseline (`random_agent.py`).

Însă, lăsat să se antreneze, agentul începe să înțeleagă regulile jocului strict din recompense. Nu a fost programat niciodată cu o regulă de tipul *„dacă vezi peretele, ferește-te”*. A învățat singur să evite pereții pentru că a fost pedepsit când s-a lovit de ei.”

## Cum funcționează (Detalii Tehnice)
„La baza agentului stă o arhitectură **Deep Q-Network (DQN)** construită în **PyTorch**, care combină Q-Learning-ul clasic cu Rețelele Neurale Artificiale. 
Sistemul funcționează astfel:
1. **Mediul (Environment)**: Jocul Snake construit în Pygame. La fiecare cadru, mediul trimite agentului **starea curentă** (11 valori boolene: dacă e pericol direct/stânga/dreapta, direcția curentă de mișcare și direcția generală spre mâncare).
2. **Recompensele (Reward Shaping)**: Acesta este semnalul prin care modelul învață:
   - `+10` puncte dacă mănâncă.
   - `-10` puncte dacă moare (lovește un perete sau propria coadă).
   - `+1` punct dacă se apropie de mâncare.
   - `-1` punct dacă se îndepărtează.
   (*Aceste mini-recompense de proximitate ajută agentul să nu se plimbe infinit în cerc și grăbesc convergența*).

3. **Rețeaua Neurală**: Agentul folosește starea de 11 intrări și încearcă să prezică **Q-values** (valoarea aștptată a calității) pentru fiecare din cele 3 mutări posibile (mers înainte, dreapta sau stânga). 

4. **Replay Memory**: Agentul reține ultimele 100.000 de stări și mutări. La finalul fiecărui joc, extrage un eșantion aleatoriu (batch) și se re-antrenează pe experiențele anterioare pentru a stabiliza învățarea și a nu uita cum să joace o situație doar pentru că nu a mai întâlnit-o recent.”

## Vizualizarea Evoluției
„Am inclus și un sistem de **live plotting** cu Matplotlib. În timp ce agentul joacă, graficul actualizează scorul per episod și media scorurilor. Veți observa o curbă clară de învățare: la început media este aproape 0, iar după ~80-100 de jocuri media crește substanțial, agentul reușind să strângă constant punctaje mari.”

## Concluzie
„Acest proiect demonstrează puterea Reinforcement Learning-ului. Un astfel de model poate fi adaptat nu doar la jocuri, ci și la probleme din viața reală: robotică, mașini autonome, sau optimizarea resurselor, fiind capabil să descopere strategii complexe, adesea superioare celor umane, pornind de la un set simplu de reguli și recompense.”

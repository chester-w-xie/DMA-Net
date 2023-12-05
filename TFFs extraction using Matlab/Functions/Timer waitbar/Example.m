% Number of iterations
N = 50;

% Creation of a timer waitbar object
TWB = Timerwaitbar(N);

% Loop
for n = 1:N
    
    % Simulation of a task 
    pause(0.1);   
    
    % Timer waitbar update
    TWB.update();
    
    % Loop break if manual cancelation
    if TWB.isinterrupted()
        break
    end
    
end

% Object deletion
TWB.delete();

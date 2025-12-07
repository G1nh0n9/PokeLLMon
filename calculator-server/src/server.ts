import express, { Request, Response } from 'express'; // Import Express and its types
// Import necessary classes and functions from the @smogon/calc library
import { calculate, Generations, Pokemon, Move, Field } from '@smogon/calc'; 

// Initialize the Express application
const app = express();
// Define the port number where the server will listen
const PORT = 3000; 

// Middleware to parse incoming JSON request bodies
app.use(express.json()); 

/**
 * @route POST /calculate
 * @description Exposes the Smogon/Showdown damage calculation function as an endpoint.
 * Receives detailed battle parameters via the request body and returns the calculation result.
 */
app.post('/calculate', (req: Request, res: Response) => {
    try {
        // Destructure necessary parameters from the request body
        const { generation, attacker, defender, move, fieldConditions } = req.body;
        
        // 1. Initialize the Generation object (e.g., Gen 9)
        const gen = Generations.get(generation);
        
        // 2. Create the Attacker Pokemon object
        const attackerObj = new Pokemon(gen, attacker.species, attacker);
        
        // 3. Create the Defender Pokemon object
        const defenderObj = new Pokemon(gen, defender.species, defender);
        
        // 4. Create the Move object
        const moveObj = new Move(gen, move);
        
        // 5. Create the Field object using conditional field conditions
        const field = fieldConditions ? new Field(fieldConditions) : new Field();
        
        // 6. Call the main calculation function from the @smogon/calc library
        const result = calculate(gen, attackerObj, defenderObj, moveObj, field);
        
        // Send the calculation result back as a JSON response
        res.json(result);
        
    } catch (error) {
        // Handle any errors during calculation or object instantiation
        console.error("Calculation error:", error);
        // Send a 500 Internal Server Error status with the error message
        res.status(500).json({ error: String(error) });
    }
});

// Start the Express server and listen on the defined port
app.listen(PORT, () => {
    // Log a success message indicating the server address
    console.log(`🚀 Damage Calculation Server running at: http://localhost:${PORT}`);
});
